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
