# kinescript-elevenlabs

[ElevenLabs](https://elevenlabs.io) voices for `kine voice`. It speaks the `text` of a scene's sounds through
ElevenLabs' API into WAV files and writes the time of each word, so layers can start on a word and captions follow
the voice. Eleven v4 and v3 act on audio tags such as `[whispers]`, `[laughs]` or `[sighs]`.

```json
{"id": "vo1", "role": "voice", "backend": "elevenlabs", "speaker": "George",
 "text": "[whispers] Nobody knows yet. [[go]][laughs] Render it.", "options": {"stability": 0.3}}
```

```bash
kine voice scene.json     # writes voice/vo1.wav and voice/vo1.words.json
kine check scene.json
```

## Install

It needs Kinescript 1.3 or newer. It has no dependencies of its own, so it installs next to Kinescript on any
Python 3.10 or newer:

```bash
pip install kinescript-elevenlabs
```

From a checkout of [Kinescript](https://github.com/drknowhow/kinescript), `pip install ./plugins/kinescript-elevenlabs`
does the same.

It needs an API key from your ElevenLabs account (Developers > API keys) in `ELEVENLABS_API_KEY`. Every line it
speaks uses characters of that account's quota. `kine voice` speaks a line again only when its text, speaker, rate,
backend or options changed, so re-running it on an unchanged scene costs nothing.

## Voices and settings

`speaker` is a voice id, or the name of a voice in your account (`George`, `Aria`); George unless the sound names one.
`rate` sets ElevenLabs' speed, which takes 0.7 to 1.2; a rate outside that is spoken at 1 and listed under
`ignored`.

`options` passes these on:

- `model`: `eleven_v4` by default. `eleven_v3` also acts on tags; other models, such as `eleven_multilingual_v2`
  or `eleven_flash_v2_5`, get the text without its tags.
- `stability`, `similarity_boost`, `style`, `use_speaker_boost`: ElevenLabs' voice settings. Lower stability gives
  a wider emotional range.
- `seed`, `language_code`, `apply_text_normalization`: passed to the request as they are.

An option it does not know is listed under `ignored` in what `kine voice` prints.

## Word times

ElevenLabs returns the time of each character it says; the plugin joins them into the words of the text as written,
leaving tags out and timing each word from its first letter to its last. When a model returns no character times,
the plugin asks ElevenLabs' forced alignment for them, which is billed like speech to text.

## Another Python

To run it from a Python other than Kinescript's, set `KINE_VOICE_CMD_ELEVENLABS` to that Python and the module:
`["/path/to/python", "-m", "kinescript_elevenlabs"]`.
