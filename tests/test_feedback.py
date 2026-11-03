"""Review notes beside a scene: saved, listed in time order, resolved, flagged stale, and read through MCP."""
import json

import pytest

from kinescript import feedback as fb
from kinescript import mcp

SCENE = {"canvas": {"w": 320, "h": 180, "fps": 30, "duration": 2, "bg": "#101820"},
         "layers": [{"id": "title", "type": "text", "text": "Hello", "x": "50%", "y": "50%", "size": 40}]}


@pytest.fixture
def s(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(SCENE), encoding="utf-8")
    return str(p)


def test_notes_list_in_time_order_with_unique_ids(s):
    fb.add(s, 1.5, "later", 0.5, 0.5, "title")
    fb.add(s, 0.2, "earlier")
    notes = fb.listing(s)
    assert [n["text"] for n in notes] == ["earlier", "later"]
    assert len({n["id"] for n in notes}) == 2
    assert notes[1]["layer"] == "title" and "x" not in notes[0]


def test_resolve_hides_note_from_open_list_and_keeps_reply(s):
    n = fb.add(s, 1, "make it bigger")
    fb.resolve(s, n["id"], "size 40 to 64")
    assert fb.listing(s) == []
    done = fb.listing(s, "resolved")[0]
    assert done["replies"][0]["text"] == "size 40 to 64"


def test_note_is_stale_after_scene_changes(s):
    n = fb.add(s, 1, "move left")
    assert fb.get(s, n["id"])["stale"] is False
    SCENE2 = dict(SCENE, layers=[])
    open(s, "w", encoding="utf-8").write(json.dumps(SCENE2))
    assert fb.get(s, n["id"])["stale"] is True


@pytest.mark.parametrize("kw", [{"t": -1, "text": "x"}, {"t": 1, "text": " "}, {"t": 1, "text": "x", "kind": "star"},
                                {"t": 1, "text": "x", "x": 2, "y": 0.5}])
def test_bad_note_is_refused(s, kw):
    with pytest.raises(fb.FeedbackError):
        fb.add(s, **kw)


def test_unknown_note_id_is_refused(s):
    with pytest.raises(fb.FeedbackError):
        fb.resolve(s, "n9")


def test_mcp_lists_and_resolves_and_marks_the_frame(s):
    n = fb.add(s, 1, "too dark", 0.5, 0.5, "title")
    out = mcp.call("kine_feedback", {"scene": s, "action": "list", "still": True})
    assert json.loads(out[0]["text"])["notes"][0]["id"] == n["id"]
    assert out[1]["type"] == "image"
    mcp.call("kine_feedback", {"scene": s, "action": "resolve", "id": n["id"], "text": "brighter"})
    assert json.loads(mcp.call("kine_feedback", {"scene": s, "action": "list"})[0]["text"])["notes"] == []


def test_mcp_rejects_unknown_note(s):
    with pytest.raises(ValueError, match="no note"):
        mcp.call("kine_feedback", {"scene": s, "action": "get", "id": "n5"})


def test_note_can_target_a_sound_word_and_carry_strokes(s):
    n = fb.add(s, 1, "mispronounced", sound="vo", word="scene")
    d = fb.add(s, 1.2, "move this", strokes=[[[0.1, 0.1], [0.4, 0.3]]], layer="title")
    assert (n["sound"], n["word"]) == ("vo", "scene") and d["kind"] == "draw" and d["strokes"][0][1] == [0.4, 0.3]


@pytest.mark.parametrize("strokes", [[], [[]], [[[0.1, 2.0]]], [[[0.1]]], "x", [[[0.1, 0.1]]] * 21])
def test_bad_strokes_are_refused(s, strokes):
    with pytest.raises(fb.FeedbackError):
        fb.add(s, 1, "x", strokes=strokes)


def test_mcp_still_draws_the_strokes(s):
    n = fb.add(s, 1, "circle it", strokes=[[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8]]])
    out = mcp.call("kine_feedback", {"scene": s, "action": "get", "id": n["id"], "still": True})
    assert out[1]["type"] == "image"
    import base64

    import skia
    img = skia.Image.MakeFromEncoded(skia.Data.MakeWithCopy(base64.b64decode(out[1]["data"])))
    px = img.toarray()  # BGRA: the yellow stroke has blue near 0 and red near 255
    h, w = px.shape[:2]
    assert px[int(h * 0.2), int(w * 0.5), 2] > 240 and px[int(h * 0.2), int(w * 0.5), 0] < 20
    assert px[int(h * 0.9), int(w * 0.1), 2] < 100


def test_mcp_still_rings_the_point(s):
    n = fb.add(s, 1, "here", 0.5, 0.5)
    out = mcp.call("kine_feedback", {"scene": s, "action": "get", "id": n["id"], "still": True})
    import base64

    import skia
    px = skia.Image.MakeFromEncoded(skia.Data.MakeWithCopy(base64.b64decode(out[1]["data"]))).toarray()
    h, w = px.shape[:2]
    ring = int(max(w, h) * 0.025)
    assert px[h // 2, w // 2 + ring, 2] > 240 and px[h // 2, w // 2 + ring, 0] < 20
