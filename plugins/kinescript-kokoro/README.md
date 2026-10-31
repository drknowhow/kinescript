# kinescript-kokoro

[Kokoro](https://github.com/hexgrad/kokoro) voices for `kine voice`. It speaks the `text` of a scene's sounds
into WAV files and writes the time of each word, so layers can start on a word and captions follow the voice.

```json
{"id": "vo1", "role": "voice", "text": "Write a scene. Check it. [[go]]Render it.", "speaker": "af_heart"}
```

```bash
kine voice scene.json     # writes voice/vo1.wav and voice/vo1.words.json
kine check scene.json
```

## Install

Kokoro needs Python 3.10 to 3.12 and PyTorch. Into the same environment as Kinescript:

```bash
pip install kinescript-kokoro
```

From a checkout of [Kinescript](https://github.com/drknowhow/kinescript), `pip install ./plugins/kinescript-kokoro`
does the same.

Kinescript itself runs on newer Pythons than Kokoro. If yours is one, install the plugin into another Python
and point `KINE_VOICE_CMD` at it; `kine voice` sends it the lines as JSON on stdin and never through a shell:

```bash
python3.12 -m venv ~/kokoro-env
~/kokoro-env/bin/pip install kinescript-kokoro
export KINE_VOICE_CMD="$HOME/kokoro-env/bin/python -m kinescript_kokoro"
```

On Windows, `set KINE_VOICE_CMD=C:\kokoro-env\Scripts\python.exe -m kinescript_kokoro`. A JSON list such as
`["C:/kokoro env/python.exe", "-m", "kinescript_kokoro"]` also works and keeps a path with spaces whole.

The first run downloads the model (327 MB, plus about 0.5 MB per voice) from Hugging Face. On the development
machine's CPU, a 4 s line took 9 s and 39 s of speech took 14 s, each including about 7 s to load the model.

## Voices and speed

`speaker` is a Kokoro voice, `af_heart` unless the sound names one. Its first letter is the language: `a`
American English, `b` British English (`af_bella`, `am_adam`, `am_michael`, `bf_emma`, `bm_george`, and the
rest of the list in Kokoro's VOICES.md). `rate` is Kokoro's speed, 1 by default.

Word times come with English voices only. Other languages are spoken, but their sounds have no words, so
word cues and voice captions do not work with them.

Two voices separated by a comma, `"speaker": "af_heart,af_bella"`, are averaged into one. Use voices of the
same language: the first letter picks it.

## Pronunciation

Kokoro reads English through [misaki](https://github.com/hexgrad/misaki), which takes a Markdown link in the
text to change how one word is said. It looks up names it doesn't know in espeak, and some come out wrong:
"Kinescript" alone is said KYNE-script.

```json
{"id": "vo1", "role": "voice", "text": "Meet [Kinescript](/kˈɪnəskɹˌɪpt/). [[go]]Render it."}
```

- `[word](/phonemes/)` says the word with the IPA phonemes between the slashes, here KIN-uh-script. `ˈ` marks
  primary stress and `ˌ` secondary; misaki's phoneme list names the symbols it accepts.
- `[word](-1)` lowers the word's stress one step, `[word](-2)` removes it, and `[word](+1)` or `[word](+2)`
  raises it.

The words file, captions and word ids keep the word as written (`Kinescript.`, id `kinescript`), and `[[go]]`
marks work around it. Keep each link to one word: misaki joins `[New York](...)` into a single timed word.

## Device

It runs on the GPU when the installed PyTorch was built for it, and on the CPU otherwise (a GPU newer than the
PyTorch build, such as an RTX 50 series card with a CUDA 12.4 build, would fail on the first line).
`KOKORO_DEVICE=cpu` or `KOKORO_DEVICE=cuda` picks one.
