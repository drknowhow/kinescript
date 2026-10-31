"""The kine command line and the MCP server: bad input gets an answer, not a traceback, and output goes where asked."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import skia

from kinescript import mcp, output
from kinescript.cli import main
from kinescript.verify import check

SCENE = {"canvas": {"w": 320, "h": 180, "fps": 30, "duration": 2, "bg": "#101820"},
         "layers": [{"id": "title", "type": "text", "text": "Hello", "x": "50%", "y": "50%", "size": 40}]}


@pytest.fixture
def s(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(SCENE), encoding="utf-8")
    return str(p)


def _env():
    env = dict(os.environ, KINE_BACKEND="cpu")
    for k in ("PYTHONUTF8", "PYTHONIOENCODING"):  # the console's own encoding, as a user's shell has it
        env.pop(k, None)
    return env


def _run(capsys, *args):
    rc = main([str(a) for a in args])
    return rc, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("args, says", [
    (["still", "{s}", "--at", "inf"], "at must be"),
    (["still", "{s}", "--at", "1", "--scale", "0"], "scale must be"),
    (["still", "{s}", "--at", "1", "--scale", "nan"], "scale must be"),
    (["still", "{s}", "--at", "1", "--scale", "100"], "largest side"),
    (["render", "{s}", "-o", "{tmp}/a.mp4", "--to", "inf"], "to must be"),
    (["render", "{s}", "-o", "{tmp}/a.mp4", "--from", "-1"], "from must be"),
    (["render", "{s}", "-o", "{tmp}/a.mp4", "--from", "5"], "past the end"),
    (["look", "{s}", "-o", "{tmp}/a.png", "--width", "100000"], "width must be"),
    (["look", "{s}", "-o", "{tmp}/a.png", "--cols", "-2"], "cols must be"),
    (["look", "{s}", "-o", "{tmp}/a.png", "-n", "0"], "count must be"),
    (["check", "{s}", "--step", "0"], "step must be"),
    (["probe", "{s}", "--at", "nan"], "at must be"),
    (["still", "{tmp}", "--at", "1"], "is a folder"),
])
def test_bad_arguments_get_an_error_not_a_traceback(capsys, tmp_path, s, args, says):
    rc, res = _run(capsys, *(a.format(s=s, tmp=tmp_path) for a in args))
    assert rc == 2 and res["ok"] is False and says in res["error"], res


@pytest.mark.parametrize("out, says", [
    ("{tmp}", "is a folder"),  # used to write the render's temp name, final.mp4, inside it
    ("{tmp}/a.mkv", "can't write .mkv"),  # used to be an MP4 under that name
    ("{tmp}/a", "no extension"),
    ("{s}", "can't write .json"),  # used to replace the scene with a video
    ("{tmp}/pic.png", "a file this scene reads"),  # used to replace its own image
])
def test_render_refuses_outputs_it_would_get_wrong(capsys, tmp_path, s, out, says):
    skia.Surface(8, 8).makeImageSnapshot().save(str(tmp_path / "pic.png"), skia.kPNG)
    spec = dict(SCENE, layers=SCENE["layers"] + [{"type": "image", "src": "pic.png", "w": 8, "h": 8}])
    (tmp_path / "s.json").write_text(json.dumps(spec), encoding="utf-8")
    before = (tmp_path / "pic.png").read_bytes()
    rc, res = _run(capsys, "render", s, "-o", out.format(tmp=tmp_path, s=s), "--to", "0.1", "--scale", "0.25")
    assert rc == 2 and says in res["error"], res
    assert json.loads((tmp_path / "s.json").read_text())["layers"] == spec["layers"]
    assert (tmp_path / "pic.png").read_bytes() == before


def test_stills_and_sheets_go_where_asked(capsys, tmp_path, s):
    rc, res = _run(capsys, "still", s, "--at", "1", "--scale", "0.25", "-o", tmp_path / "new" / "dir" / "a.png")
    assert rc == 0 and (tmp_path / "new" / "dir" / "a.png").exists()  # said ok and wrote nothing, before
    rc, res = _run(capsys, "still", s, "--at", "1", "-o", tmp_path / "a.webp")
    assert rc == 2 and "can't write .webp" in res["error"]  # was JPEG bytes in a .webp
    rc, res = _run(capsys, "look", s, "-n", "2", "--width", "64", "-o", tmp_path / "sheet" / "s.jpg")
    assert rc == 0 and (tmp_path / "sheet" / "s.jpg").read_bytes()[:2] == b"\xff\xd8"  # was PNG bytes in a .jpg
    rc, res = _run(capsys, "look", s, "-n", "3", "--cols", "1000", "--width", "64", "-o", tmp_path / "c.png")
    assert res["grid"] == [3, 1]  # was [1000, 1]: a sheet 1000 thumbnails wide
    rc, res = _run(capsys, "look", s, "--at", "end", "-0.5", "--width", "64", "-o", tmp_path / "t.png")
    assert res["times"] == [1.967, 1.5]  # as still reads them; "end" was refused and -0.5 drew a blank frame


def test_still_names_times_from_the_end_apart(capsys, tmp_path, s):
    for at in ("0.5", "-0.5", "end"):
        assert main(["still", s, "--at", at, "--scale", "0.1"]) == 0
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["s_0.5s.png", "s_end-0.5s.png", "s_end.png"]


def test_a_png_sequence_pattern_numbers_the_files(capsys, tmp_path, s):
    rc, res = _run(capsys, "render", s, "-o", tmp_path / "seq" / "f_%03d.png", "--to", "0.1", "--scale", "0.1")
    assert rc == 0 and sorted(p.name for p in (tmp_path / "seq").iterdir()) == ["f_000.png", "f_001.png", "f_002.png"]


def test_check_sees_a_layer_shorter_than_its_step():
    flash = {"id": "flash", "type": "text", "text": "Offscreen flash text here", "x": "95%", "y": "50%", "size": 60,
             "at": 0.3, "dur": 0.15}  # between the samples at 0.25 and 0.5
    res = check({"canvas": {"w": 640, "h": 360, "fps": 30, "duration": 2}, "layers": [flash]})
    assert "text-offscreen" in [i["code"] for i in res["issues"]], res


def test_cli_output_is_utf8_in_a_pipe(tmp_path):
    spec = {"canvas": {"w": 320, "h": 180, "duration": 1},
            "layers": [{"id": "标题🚀", "type": "text", "text": "café 你好", "x": "50%", "y": "50%", "size": 30}]}
    (tmp_path / "u.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "kinescript.cli", "probe", str(tmp_path / "u.json"), "--at", "0.5"],
                       capture_output=True, env=_env(), timeout=120)
    # On Windows a pipe's default code page could not hold the id: rc 2 and a 'charmap' error after the work was done
    assert p.returncode == 0, p.stdout + p.stderr
    assert "标题🚀" in json.loads(p.stdout.decode("utf-8"))["visible"]


def test_mcp_server_survives_lines_that_are_not_requests():
    probe = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "kine_probe", "arguments": {
        "scene": {"canvas": {"w": 64, "h": 36, "duration": 1}, "layers": [{"id": "café 你好", "type": "rect"}]}}}}
    lines = [b"123", b'"hello"', b'[{"jsonrpc":"2.0","id":1,"method":"ping"}]',
             b'{"jsonrpc":"2.0","id":2,"method":"initialize","params":null}',
             b'{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"kine_docs","arguments":[1]}}',
             json.dumps(probe, ensure_ascii=False).encode("utf-8"),  # raw UTF-8, as Node's JSON.stringify sends
             b'{"jsonrpc":"2.0","id":9,"method":"ping"}']
    p = subprocess.run([sys.executable, "-m", "kinescript.cli", "mcp"], input=b"\n".join(lines) + b"\n",
                       capture_output=True, env=_env(), timeout=120)
    replies = [json.loads(line) for line in p.stdout.decode("utf-8").splitlines()]
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    assert [r.get("error", {}).get("code") for r in replies[:3]] == [-32600] * 3  # each of these ended the server
    by_id = {r["id"]: r for r in replies[3:]}
    assert by_id[2]["result"]["serverInfo"]["name"] == "kinescript"
    assert by_id[4]["error"]["code"] == -32602
    layers = json.loads(by_id[3]["result"]["content"][0]["text"])["layers"]
    assert [L["id"] for L in layers] == ["café 你好"]  # arrived as cafÃ©
    assert by_id[9]["result"] == {}


def test_mcp_check_reports_a_scene_that_fails_to_load_as_issues():
    bad = {"canvas": {"w": 320, "h": 180}, "layers": [{"id": "a", "type": "nonsense"}]}
    res = json.loads(mcp.call("kine_check", {"scene": bad})[0]["text"])  # was a one-line isError string
    assert res["ok"] is False and [i["code"] for i in res["issues"]][:1] == ["bad-type"]


def test_mcp_arguments_are_checked_and_still_takes_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='missing argument "scene"'):
        mcp.call("kine_check", {})
    with pytest.raises(ValueError, match="scene must be"):
        mcp.call("kine_check", {"scene": 5})
    with pytest.raises(ValueError, match="quality must be"):
        mcp.call("kine_render", {"scene": SCENE, "out": "a.mp4", "quality": "ultra"})
    with pytest.raises(ValueError, match="from must be"):
        mcp.call("kine_render", {"scene": SCENE, "out": "a.mp4", "from": "soon"})
    image = mcp.call("kine_still", {"scene": SCENE, "at": "end", "scale": 0.1})[0]  # "end" was not a number
    assert image["type"] == "image" and not (Path(tempfile.gettempdir()) / f"kine_still_{os.getpid()}.png").exists()
    res = json.loads(mcp.call("kine_render", {"scene": SCENE, "out": "a.mp4", "to": 0.1, "scale": 0.25})[0]["text"])
    assert Path(res["out"]).is_absolute() and Path(res["out"]).samefile(tmp_path / "a.mp4")


def test_watch_waits_out_a_missing_or_locked_file_and_stops_on_ctrl_c(monkeypatch, capsys, tmp_path, s):
    from kinescript import verify
    drawn, ticks = [], iter(range(10))

    def sheet(src, out, count=12):
        drawn.append(src)
        if len(drawn) == 1:
            raise PermissionError("locked by the editor")

    def sleep(_):
        k = next(ticks)
        if k == 1:
            os.remove(s)  # an editor saving by delete and rename
        elif k == 2:
            (tmp_path / "s.json").write_text(json.dumps(SCENE), encoding="utf-8")
            os.utime(s, (time.time() + 5, time.time() + 5))
        elif k == 4:
            raise KeyboardInterrupt
    monkeypatch.setattr(verify, "contact_sheet", sheet)
    monkeypatch.setattr(time, "sleep", sleep)
    assert main(["watch", s]) == 0
    assert len(drawn) == 3 and "stopped" in capsys.readouterr().err  # locked, read again, then the new save


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_warm_render_workers_exit_when_their_parent_is_killed():
    script = ("import os, time\nfrom kinescript import output\n"
              "print(output._pool(1).submit(os.getpid).result(), flush=True)\ntime.sleep(60)\n")
    p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, env=_env())
    worker = int(p.stdout.readline())
    p.kill()  # as when an MCP client kills the server: the pool's idle timer dies with it
    p.wait()
    try:
        end = time.time() + 15
        while _alive(worker) and time.time() < end:
            time.sleep(0.2)
        assert not _alive(worker)
    finally:
        if _alive(worker):
            os.kill(worker, 9 if sys.platform != "win32" else 1)
