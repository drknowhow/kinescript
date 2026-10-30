"""For KINE_VOICE_CMD: a JSON list of requests on stdin, the list of results on stdout.

    KINE_VOICE_CMD='["/path/to/python", "-m", "kinescript_kokoro"]' kine voice scene.json
"""
import contextlib
import json
import sys

from . import synth

reqs = json.loads(sys.stdin.read())
with contextlib.redirect_stdout(sys.stderr):  # Kokoro's own messages stay out of the answer
    results = [synth(r) for r in reqs]
sys.stdout.write(json.dumps(results))
