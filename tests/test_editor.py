"""kine edit: frames, layer hits and notes over HTTP, locked to this machine."""
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from kinescript import editor, feedback

SCENE = {"canvas": {"w": 320, "h": 180, "fps": 30, "duration": 2, "bg": "#101820"},
         "layers": [{"id": "back", "type": "rect", "x": 0, "y": 0, "w": 320, "h": 180, "anchor": "topleft", "fill": "#223"},
                    {"id": "title", "type": "text", "text": "Hello", "x": "50%", "y": "50%", "size": 40}]}


@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("KINE_BACKEND", "cpu")
    p = tmp_path / "s.json"
    p.write_text(json.dumps(SCENE), encoding="utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), editor.make_handler(p))
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield httpd.server_port, p
    httpd.shutdown()
    httpd.server_close()


def req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    h = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body)
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, data, h)
    r = c.getresponse()
    out = (r.status, r.read(), dict(r.getheaders()))
    c.close()
    return out


def test_page_and_scene_info_load(srv):
    port, _ = srv
    status, body, _ = req(port, "GET", "/")
    assert status == 200 and b"Kinescript editor" in body
    info = json.loads(req(port, "GET", "/scene")[1])
    assert (info["w"], info["h"], info["duration"], info["video"]) == (320, 180, 2.0, None)


def test_frame_is_a_png(srv):
    port, _ = srv
    status, body, hdr = req(port, "GET", "/frame?t=1&scale=0.5")
    assert status == 200 and hdr["Content-Type"] == "image/png" and body[:4] == b"\x89PNG"


def test_hit_returns_the_topmost_layer_at_the_point(srv):
    port, _ = srv
    centre = json.loads(req(port, "GET", "/hit?x=0.5&y=0.5&t=1")[1])
    corner = json.loads(req(port, "GET", "/hit?x=0.02&y=0.02&t=1")[1])
    assert centre["layer"] == "title" and corner["layer"] == "back"


@pytest.mark.parametrize("path", ["/frame?t=abc", "/frame?t=nan", "/hit?x=0.5&y=0.5", "/hit?x=0.5&y=0.5&t=-1"])
def test_bad_query_is_a_400_or_clamped_not_a_crash(srv, path):
    port, _ = srv
    status, body, _ = req(port, "GET", path)
    assert status in (200, 400) and (status == 200 or "error" in json.loads(body))


def test_notes_round_trip_and_resolve(srv):
    port, scene = srv
    status, body, _ = req(port, "POST", "/notes", {"t": 1, "x": 0.5, "y": 0.5, "layer": "title", "text": "bigger"})
    assert status == 201
    nid = json.loads(body)["note"]["id"]
    assert [n["id"] for n in json.loads(req(port, "GET", "/notes")[1])["notes"]] == [nid]
    assert feedback.listing(scene)[0]["text"] == "bigger"
    req(port, "POST", "/notes", {"action": "resolve", "id": nid})
    assert json.loads(req(port, "GET", "/notes")[1])["notes"] == []
    req(port, "POST", "/notes", {"action": "reopen", "id": nid})
    assert len(json.loads(req(port, "GET", "/notes")[1])["notes"]) == 1


def test_empty_note_is_refused(srv):
    port, scene = srv
    status, _, _ = req(port, "POST", "/notes", {"t": 1, "text": "  "})
    assert status == 400 and feedback.listing(scene) == []


def test_post_without_json_content_type_is_refused(srv):
    port, _ = srv
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/notes", "t=1", {"Content-Type": "text/plain"})
    assert c.getresponse().status == 400


def test_foreign_host_header_is_refused(srv):
    port, _ = srv
    status, _, _ = req(port, "GET", "/scene", headers={"Host": "evil.example"})
    assert status == 403


def test_video_serves_byte_ranges(srv):
    port, scene = srv
    assert req(port, "GET", "/video")[0] == 404
    scene.with_suffix(".mp4").write_bytes(bytes(range(200)))
    status, body, hdr = req(port, "GET", "/video", headers={"Range": "bytes=10-19"})
    assert status == 206 and body == bytes(range(10, 20)) and hdr["Content-Range"] == "bytes 10-19/200"
    assert req(port, "GET", "/video", headers={"Range": "bytes=500-"})[0] == 416


def test_scene_that_no_longer_loads_reports_an_error(srv):
    port, scene = srv
    scene.write_text("{ not json", encoding="utf-8")
    status, body, _ = req(port, "GET", "/frame?t=0")
    assert status == 500 and "error" in json.loads(body)


def test_scene_lists_layers_with_timing_and_controls(srv):
    port, _ = srv
    layers = {l["id"]: l for l in json.loads(req(port, "GET", "/scene")[1])["layers"]}
    assert layers["title"]["type"] == "text" and layers["title"]["in"] == 0
    keys = [c["key"] for c in layers["title"]["controls"]]
    assert {"text", "x", "size"} <= set(keys) and "r" not in keys
    assert next(c for c in layers["title"]["controls"] if c["key"] == "size")["editable"]


def test_hit_returns_boxes_for_visible_layers(srv):
    port, _ = srv
    boxes = json.loads(req(port, "GET", "/hit?x=0&y=0&t=1")[1])["boxes"]
    assert set(boxes) == {"back", "title"} and len(boxes["title"]) == 4


def test_prop_edit_rewrites_the_scene_and_logs_a_note(srv):
    port, scene = srv
    h = json.loads(req(port, "GET", "/scene")[1])["hash"]
    status, body, _ = req(port, "POST", "/prop", {"layer": "title", "key": "size", "value": 80, "hash": h, "t": 1})
    assert status == 200 and json.loads(body)["hash"] != h
    assert json.loads(scene.read_text(encoding="utf-8"))["layers"][1]["size"] == 80
    note = feedback.listing(scene)[0]
    assert note["kind"] == "prop" and note["change"] == {"size": {"from": 40, "to": 80}} and note["layer"] == "title"


@pytest.mark.parametrize("body, why", [
    ({"layer": "title", "key": "size", "value": "huge"}, "breaks the scene"),
    ({"layer": "title", "key": "id", "value": "x"}, "cannot be edited"),
    ({"layer": "nope", "key": "size", "value": 5}, "no layer"),
    ({"layer": "title", "key": "size", "value": True}, "number or text"),
])
def test_bad_prop_edit_is_refused_and_leaves_the_file_alone(srv, body, why):
    port, scene = srv
    before = scene.read_bytes()
    body["hash"] = json.loads(req(port, "GET", "/scene")[1])["hash"]
    status, out, _ = req(port, "POST", "/prop", body)
    assert status == 400 and why in json.loads(out)["error"]
    assert scene.read_bytes() == before and feedback.listing(scene) == []


def test_prop_edit_refuses_a_stale_page(srv):
    port, scene = srv
    status, out, _ = req(port, "POST", "/prop", {"layer": "title", "key": "size", "value": 80, "hash": "old"})
    assert status == 400 and "reload" in json.loads(out)["error"]


def test_prop_edit_will_not_rewrite_a_scene_with_comments(srv):
    port, scene = srv
    scene.write_text('// keep me\n' + json.dumps(SCENE), encoding="utf-8")
    status, out, _ = req(port, "POST", "/prop", {"layer": "title", "key": "size", "value": 80,
                                                   "hash": feedback.scene_hash(scene)})
    assert status == 400 and "comments" in json.loads(out)["error"] and scene.read_text(encoding="utf-8").startswith("// keep me")


def test_prop_edit_can_move_a_layer_in_one_save(srv):
    port, scene = srv
    h = json.loads(req(port, "GET", "/scene")[1])["hash"]
    status, _, _ = req(port, "POST", "/prop", {"layer": "title", "changes": {"x": 100, "y": 50}, "hash": h, "t": 0})
    assert status == 200
    title = json.loads(scene.read_text(encoding="utf-8"))["layers"][1]
    assert (title["x"], title["y"]) == (100, 50) and len(feedback.listing(scene)) == 1


def test_hit_returns_layer_positions_in_pixels(srv):
    port, _ = srv
    pos = json.loads(req(port, "GET", "/hit?x=0&y=0&t=1")[1])["pos"]
    assert pos["title"] == [160.0, 90.0]


@pytest.fixture
def vsrv(tmp_path, monkeypatch):
    import wave
    monkeypatch.setenv("KINE_BACKEND", "cpu")
    with wave.open(str(tmp_path / "vo.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000)
    p = tmp_path / "v.json"
    p.write_text(json.dumps({**SCENE, "audio": [{"id": "vo", "role": "voice", "text": "Hi there", "src": "vo.wav"}],
                             "layers": SCENE["layers"] + [{"id": "cc", "type": "captions", "cues": [[0, 1, "Hello"]]}]}),
                 encoding="utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), editor.make_handler(p))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_port, p
    httpd.shutdown()
    httpd.server_close()


def test_audio_serves_a_sound_file_with_ranges(vsrv):
    port, scene = vsrv
    status, body, hdr = req(port, "GET", "/audio?id=vo", headers={"Range": "bytes=0-43"})
    assert status == 206 and body[:4] == b"RIFF" and len(body) == 44 and "audio/wav" in hdr["Content-Type"]
    assert req(port, "GET", "/audio?id=nope")[0] == 404
    assert req(port, "GET", "/audio")[0] == 404


def test_scene_lists_sounds_and_caption_cues(vsrv):
    port, _ = vsrv
    info = json.loads(req(port, "GET", "/scene")[1])
    vo = info["sounds"][0]
    assert (vo["id"], vo["role"], vo["speaks"]) == ("vo", "voice", True)
    assert "text" in [c["key"] for c in vo["controls"]]
    cc = next(l for l in info["layers"] if l["id"] == "cc")
    assert cc["cues"] == [[0.0, 1.0, "Hello"]]


def test_sound_text_edit_is_written_and_flagged_for_kine_voice(vsrv):
    port, scene = vsrv
    h = json.loads(req(port, "GET", "/scene")[1])["hash"]
    status, _, _ = req(port, "POST", "/prop", {"sound": "vo", "changes": {"text": "Hello world"}, "hash": h, "t": 0.5})
    assert status == 200
    assert json.loads(scene.read_text(encoding="utf-8"))["audio"][0]["text"] == "Hello world"
    note = feedback.listing(scene)[0]
    assert note["sound"] == "vo" and "kine_voice" in note["text"] and "layer" not in note


def test_sound_edit_refuses_layer_only_keys(vsrv):
    port, scene = vsrv
    h = json.loads(req(port, "GET", "/scene")[1])["hash"]
    status, out, _ = req(port, "POST", "/prop", {"sound": "vo", "changes": {"size": 5}, "hash": h})
    assert status == 400 and "cannot be edited" in json.loads(out)["error"]


def test_notes_post_accepts_word_cue_and_strokes(vsrv):
    port, scene = vsrv
    req(port, "POST", "/notes", {"t": 0.5, "text": "too fast", "sound": "vo", "word": "there", "layer": "cc"})
    req(port, "POST", "/notes", {"t": 0.5, "text": "typo", "layer": "cc", "word": "Hello"})
    req(port, "POST", "/notes", {"t": 1, "text": "this bit", "strokes": [[[0.1, 0.1], [0.5, 0.5]]]})
    status, _, _ = req(port, "POST", "/notes", {"t": 1, "text": "bad", "strokes": [[[3, 3]]]})
    notes = {n["text"]: n for n in feedback.listing(scene)}
    assert status == 400 and notes["too fast"]["word"] == "there" and notes["typo"]["layer"] == "cc"
    assert notes["this bit"]["kind"] == "draw"


def test_hide_removes_a_layer_from_the_frame_and_from_hits_without_touching_the_file(srv):
    port, scene = srv
    before = scene.read_bytes()
    assert json.loads(req(port, "GET", "/hit?x=0.5&y=0.5&t=1")[1])["layer"] == "title"
    hit = json.loads(req(port, "GET", "/hit?x=0.5&y=0.5&t=1&hide=title")[1])
    assert hit["layer"] == "back" and "title" not in hit["layers"]
    shown = req(port, "GET", "/frame?t=1")[1]
    hidden = req(port, "GET", "/frame?t=1&hide=title")[1]
    assert shown != hidden and hidden != req(port, "GET", "/frame?t=1&hide=title,back")[1]
    assert scene.read_bytes() == before


def test_hide_ignores_unknown_ids(srv):
    port, _ = srv
    assert req(port, "GET", "/frame?t=1&hide=nope")[1] == req(port, "GET", "/frame?t=1")[1]


def _serve(tmp_path, scene):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(scene), encoding="utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), editor.make_handler(p))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_hit_works_when_the_scene_has_captions(vsrv):
    # captions record a box with no anchor; /hit used to fail on that, which made nothing selectable
    port, _ = vsrv
    status, body, _ = req(port, "GET", "/hit?x=0&y=0&t=0.5")
    data = json.loads(body)
    assert status == 200, data
    assert "title" in data["boxes"]


def test_group_gets_a_box_under_its_children_and_every_visible_layer_has_one(tmp_path, monkeypatch):
    monkeypatch.setenv("KINE_BACKEND", "cpu")
    scene = {**SCENE, "layers": [{"id": "grp", "type": "group", "x": 20, "y": 20, "children": [
        {"id": "a", "type": "text", "text": "One", "x": 10, "y": 30, "size": 24, "anchor": "left"},
        {"id": "b", "type": "text", "text": "Two", "x": 10, "y": 80, "size": 24, "anchor": "left"}]}]}
    httpd = _serve(tmp_path, scene)
    try:
        data = json.loads(req(httpd.server_port, "GET", "/hit?x=0&y=0&t=1")[1])
        assert {"grp", "a", "b"} <= set(data["boxes"])
        assert data["layers"].index("grp") < data["layers"].index("a")  # the group lies under its children
        ga, a = data["boxes"]["grp"], data["boxes"]["a"]
        assert ga[0] <= a[0] and ga[1] <= a[1] and ga[2] >= a[2] and ga[3] >= a[3]
        mid = [(a[0] + a[2]) / 2, (a[1] + a[3]) / 2]
        assert json.loads(req(httpd.server_port, "GET", f"/hit?x={mid[0]}&y={mid[1]}&t=1")[1])["layer"] == "a"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_layers_shown_as_a_texture_borrow_the_box_of_the_layer_that_shows_them(tmp_path, monkeypatch):
    monkeypatch.setenv("KINE_BACKEND", "cpu")
    scene = {**SCENE, "layers": [
        {"id": "app", "type": "group", "w": 200, "h": 100, "children": [
            {"id": "app-title", "type": "text", "text": "Hi", "x": 100, "y": 50, "size": 30}]},
        {"id": "screen", "type": "model", "shape": "plane", "w": 200, "h": 100, "texture": "#app", "x": "50%", "y": "50%"}]}
    httpd = _serve(tmp_path, scene)
    try:
        data = json.loads(req(httpd.server_port, "GET", "/hit?x=0&y=0&t=1")[1])
        assert "screen" in data["boxes"], data
        assert {"app", "app-title"} <= set(data["approx"])
        assert data["boxes"]["app-title"] == data["boxes"]["screen"]
        assert data["layers"].index("app-title") < data["layers"].index("screen")  # clicking still picks the screen
    finally:
        httpd.shutdown()
        httpd.server_close()
