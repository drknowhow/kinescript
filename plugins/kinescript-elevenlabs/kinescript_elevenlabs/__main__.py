"""For KINE_VOICE_CMD_ELEVENLABS: a JSON list of requests on stdin, the list of results on stdout.

    KINE_VOICE_CMD_ELEVENLABS='["/path/to/python", "-m", "kinescript_elevenlabs"]' kine voice scene.json
"""
import json
import sys

from . import synth

sys.stdout.write(json.dumps([synth(r) for r in json.loads(sys.stdin.read())]))
