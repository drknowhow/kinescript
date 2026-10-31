"""For KINE_VOICE_CMD_CHATTERBOX: a JSON list of requests on stdin, the list of results on stdout.

    KINE_VOICE_CMD_CHATTERBOX='["/path/to/python", "-m", "kinescript_chatterbox"]' kine voice scene.json
"""
import contextlib
import json
import sys

from . import synth

reqs = json.loads(sys.stdin.read())
with contextlib.redirect_stdout(sys.stderr):  # Chatterbox's own messages stay out of the answer
    results = [synth(r) for r in reqs]
sys.stdout.write(json.dumps(results))
