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
    try:  # exited but not yet reaped by whoever adopted it
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
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


def test_times_and_ranges_past_the_end_are_noted(capsys, tmp_path, s):
    rc, res = _run(capsys, "render", s, "--to", "99", "--scale", "0.25", "-o", tmp_path / "a.mp4")
    assert rc == 0 and res["frames"] == 60 and "to 99s is past the end of the 2s scene" in res["notes"][0]  # silent
    rc, res = _run(capsys, "render", s, "--to", "2", "--scale", "0.25", "-o", tmp_path / "b.mp4")
    assert res["frames"] == 60 and "notes" not in res  # the end itself is not past it
    rc, res = _run(capsys, "look", s, "--at", "99", "1", "-100", "--width", "64", "-o", tmp_path / "c.png")
    assert res["times"] == [1.967, 1.0, 0.0] and len(res["notes"]) == 2, res  # clamped and said nothing
    assert "at 99s is past the end" in res["notes"][0] and "at -100s is before the start" in res["notes"][1]
    rc, res = _run(capsys, "still", s, "--at", "99", "--scale", "0.1", "-o", tmp_path / "d.png")
    assert res["frame"] == 59 and "drew the last frame" in res["notes"][0]
    rc, res = _run(capsys, "still", s, "--at", "end", "--scale", "0.1", "-o", tmp_path / "e.png")
    assert res["frame"] == 59 and "notes" not in res


@pytest.mark.parametrize("args, says", [
    (["still", "{s}", "--at", "1", "--scale", "0.0001"], "each side must be 2 px or more"),  # was a 2x2 image
    (["render", "{s}", "-o", "{tmp}/a.mp4", "--scale", "0.001"], "each side must be 2 px or more"),  # a 2x2 video
    (["render", "{s}", "-o", "{tmp}/a.mp4", "--from", "1", "--to", "1"], "to must be after from"),
    (["render", "{s}", "-o", "{tmp}/a.mp4", "--to", "0"], "to must be after from"),
    (["render", "{s}", "-o", "{tmp}/a.mp4", "--from", "2"], "past the end of the 2s scene; its last frame starts at"),
])
def test_bad_ranges_and_scales_say_what_is_wrong(capsys, tmp_path, s, args, says):
    rc, res = _run(capsys, *(a.format(s=s, tmp=tmp_path) for a in args))
    assert rc == 2 and says in res["error"], res


def test_from_inside_the_last_frame_renders_it(capsys, tmp_path):
    p = tmp_path / "three.json"
    p.write_text(json.dumps(dict(SCENE, canvas=dict(SCENE["canvas"], duration=3))), encoding="utf-8")
    rc, res = _run(capsys, "render", p, "--from", "2.99", "--scale", "0.25", "-o", tmp_path / "a.mp4")
    assert rc == 0 and res["frames"] == 1, res  # 2.99 rounded to 3 s: "nothing to render"
    rc, res = _run(capsys, "render", p, "--from", "1.01", "--to", "1.02", "--scale", "0.25", "-o", tmp_path / "b.mp4")
    assert rc == 0 and res["frames"] == 1, res  # the frame on screen from 1.0 s to 1.033 s
    rc, res = _run(capsys, "render", p, "--from", "0.5", "--to", "1", "-o", tmp_path / "seq" / "f_%03d.png")
    assert res["frames"] == 15 and sorted(q.name for q in (tmp_path / "seq").iterdir())[0] == "f_015.png"


def _gif_frames(path) -> int:
    from kinescript import media
    return int(subprocess.run([media.FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                               "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True,
                              text=True, check=True).stdout.strip())


def test_a_gif_reports_its_own_frames(capsys, tmp_path, s):
    rc, res = _run(capsys, "render", s, "--scale", "0.25", "-o", tmp_path / "a.gif")
    assert rc == 0 and res["fps"] == 15 and res["frames"] == 30 == _gif_frames(tmp_path / "a.gif")  # said 60
    fast = tmp_path / "fast.json"
    fast.write_text(json.dumps(dict(SCENE, canvas=dict(SCENE["canvas"], fps=60))), encoding="utf-8")
    rc, res = _run(capsys, "render", fast, "--to", "0.01", "--scale", "0.25", "-o", tmp_path / "one.gif")
    assert rc == 0 and res["frames"] == 1 == _gif_frames(tmp_path / "one.gif"), res  # ffmpeg failed on no frame


@pytest.mark.parametrize("duration, frames", [(2.45, 74), (2.55, 77), (0.49, 15), (1.0, 30), (0.01, 1)])
def test_the_frame_count_rounds_half_a_frame_up(duration, frames):
    from kinescript.scene import Scene
    assert Scene(dict(SCENE, canvas=dict(SCENE["canvas"], duration=duration))).frames == frames  # 2.55 s gave 76


def test_workers_are_capped_and_the_number_used_is_reported(capsys, tmp_path, s, monkeypatch):
    monkeypatch.setenv("KINE_BACKEND", "cpu")
    rc, res = _run(capsys, "render", s, "-j", "100", "--to", "0.1", "--scale", "0.25", "-o", tmp_path / "a.mp4")
    assert rc == 0 and res["workers"] == 3, res  # -j 100 on 60 frames started 60 processes
    assert "workers 100 is more than this render can use; it used 3" in res["notes"][0]
    assert output._cap_workers(100, 60, 8, []) == 8 and output._cap_workers(4, 60, 8, []) == 4


def test_kine_backend_must_be_one_it_knows(capsys, tmp_path, s, monkeypatch):
    from kinescript.render import _GL, backend_setting, gpu_context
    monkeypatch.setenv("KINE_BACKEND", "bogus")  # was the GPU
    for args in (["still", s, "--at", "1", "-o", tmp_path / "a.png"], ["render", s, "-o", tmp_path / "a.mp4"],
                 ["doctor"]):
        rc, res = _run(capsys, *args)
        assert rc == 2 and "KINE_BACKEND='bogus' is not a backend: use cpu, gpu" in res["error"], res
    monkeypatch.setenv("KINE_BACKEND", " CPU ")
    assert backend_setting() == "cpu"
    monkeypatch.setenv("KINE_BACKEND", "gpu")
    monkeypatch.setitem(_GL, "ctx", None)  # as on a machine with no GPU
    with pytest.raises(RuntimeError, match="KINE_BACKEND=gpu, but no OpenGL context"):
        gpu_context()


def test_a_missing_ffmpeg_or_ffprobe_is_named(capsys, tmp_path, s, monkeypatch):
    import wave
    from kinescript import media
    with wave.open(str(tmp_path / "tone.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\0\0" * 800)
    (tmp_path / "sound.json").write_text(json.dumps(dict(SCENE, audio=[{"src": "tone.wav"}])), encoding="utf-8")
    monkeypatch.setattr(media, "FFMPEG", "kine-test-no-ffmpeg")
    monkeypatch.setattr(media, "FFPROBE", "kine-test-no-ffprobe")
    rc, res = _run(capsys, "render", s, "--to", "0.1", "-o", tmp_path / "a.mp4")
    assert rc == 2 and res["error"].startswith("ffmpeg and ffprobe are not on the PATH"), res  # was [WinError 2]
    assert "winget install Gyan.FFmpeg" in res["error"]
    rc, res = _run(capsys, "probe", tmp_path / "tone.wav")
    assert rc == 2 and "ffprobe are not on the PATH" in res["error"], res
    rc, res = _run(capsys, "check", tmp_path / "sound.json")
    assert rc == 1 and "ffprobe are not on the PATH" in res["issues"][0]["message"], res
    rc, res = _run(capsys, "doctor")
    assert rc == 1 and "install ffmpeg" in res["error"] and res["ffmpeg"] is None
    mcp._handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                 "params": {"name": "kine_probe", "arguments": {"scene": str(tmp_path / "tone.wav")}}})
    reply = json.loads(capsys.readouterr().out)["result"]
    assert reply["isError"] and "ffprobe are not on the PATH" in reply["content"][0]["text"]


def test_a_voice_command_that_fails_says_which_variable(capsys, tmp_path, monkeypatch):
    from kinescript import voice
    monkeypatch.setattr(voice, "backends", lambda: {})
    p = tmp_path / "v.json"
    p.write_text(json.dumps(dict(SCENE, audio=[{"id": "vo", "role": "voice", "text": "Hello there"}])), encoding="utf-8")
    tool = tmp_path / "say.py"
    tool.write_text("print('not json')\n")
    for cmd, says in [
        ("kine-test-no-such-program", "KINE_VOICE_CMD could not start kine-test-no-such-program"),  # [WinError 2]
        (json.dumps([sys.executable, str(tool)]), "printed 'not json', not JSON; it must print a JSON list"),
        ('["python", "-m"', "KINE_VOICE_CMD is not a command"),
        ("[]", "KINE_VOICE_CMD is not a command: '[]' names no program"),
    ]:
        monkeypatch.setenv("KINE_VOICE_CMD", cmd)
        rc, res = _run(capsys, "voice", p)
        assert rc == 2 and says in res["error"], res
    tool.write_text("import sys; sys.stdin.read(); print('[]')\n")
    monkeypatch.setenv("KINE_VOICE_CMD", json.dumps([sys.executable, str(tool)]))
    rc, res = _run(capsys, "voice", p)
    assert rc == 2 and "must print a JSON list of 1 result object, one per request" in res["error"], res


def test_exit_codes_tell_a_crash_from_a_lint_error(capsys, tmp_path, s, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"canvas": {"w": 320, "h": 180}, "layers": [{"type": "nonsense"}]}), encoding="utf-8")
    assert _run(capsys, "check", bad)[0] == 1  # check found errors
    assert _run(capsys, "still", s, "--at", "nan")[0] == 2  # a bad argument

    def boom(*a, **kw):
        return 1 / 0
    monkeypatch.setattr(output, "render", boom)
    rc, res = _run(capsys, "render", s, "-o", tmp_path / "a.mp4")
    assert rc == 3, res  # was 1, the code check uses for a scene with errors, with a traceback
    assert res["error"].startswith("internal error: ZeroDivisionError: division by zero at cli.py:")
    assert "This is a bug in Kinescript; please report it at https://github.com/drknowhow/kinescript/issues" \
        in res["error"]
    monkeypatch.setenv("KINE_DEBUG", "1")
    assert main(["render", s]) == 3 and "Traceback" in capsys.readouterr().err

    def stop(*a, **kw):
        raise KeyboardInterrupt
    monkeypatch.setattr(output, "render", stop)
    assert main(["render", s]) == 130


def test_new_makes_folders_and_mcp_leaves_no_temporary_images(capsys, tmp_path, s):
    rc, res = _run(capsys, "new", tmp_path / "sub" / "dir" / "x.json")
    assert rc == 0 and (tmp_path / "sub" / "dir" / "x.json").is_file()  # [Errno 2] before
    rc, res = _run(capsys, "render", s, "--to", "0.1", "--scale", "0.25", "-o", tmp_path / "new" / "dir" / "a.mp4")
    assert rc == 0 and (tmp_path / "new" / "dir" / "a.mp4").is_file()  # as render, still and look all do
    sheet = mcp.call("kine_look", {"scene": SCENE, "count": 2, "width": 64, "times": [99]})
    assert sheet[0]["type"] == "image" and "past the end" in json.loads(sheet[1]["text"])["notes"][0]
    image, info = mcp.call("kine_still", {"scene": SCENE, "at": 99, "scale": 0.1})
    assert image["type"] == "image" and json.loads(info["text"])["frame"] == 59
    assert "drew the last frame" in json.loads(info["text"])["notes"][0]
    tmp = Path(tempfile.gettempdir())
    assert not (tmp / f"kine_look_{os.getpid()}.png").exists() and not (tmp / f"kine_still_{os.getpid()}.png").exists()


def test_a_scene_that_is_not_utf8_is_named(capsys, tmp_path):
    p = tmp_path / "utf16.json"
    p.write_text(json.dumps(SCENE), encoding="utf-16")
    rc, res = _run(capsys, "check", p)
    assert rc == 1 and "utf16.json is not UTF-8 text" in res["issues"][0]["message"], res
    rc, res = _run(capsys, "render", p, "-o", tmp_path / "a.mp4")
    assert rc == 2 and "utf16.json is not UTF-8 text" in res["error"], res  # 'utf-8' codec can't decode byte 0xff
