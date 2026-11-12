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

On Python 3.10 to 3.12, into the same environment as Kinescript (this brings PyTorch):

```bash
pip install kinescript-kokoro
```

On Python 3.13, the plugin runs the same model on onnxruntime instead, with no PyTorch. misaki, the library that
turns Kokoro's English text into phonemes, declares Python below 3.13 but runs on it, so pip needs a flag:

```bash
pip install --ignore-requires-python kinescript-kokoro
```

From a checkout of [Kinescript](https://github.com/drknowhow/kinescript), `pip install ./plugins/kinescript-kokoro`
does the same (with the flag on 3.13).

Or install the plugin into another Python and point `KINE_VOICE_CMD` at it; `kine voice` sends it the lines as
JSON on stdin and never through a shell:

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
"Kokoro" alone is said ko-KOR-oh.

```json
{"id": "vo1", "role": "voice", "text": "Meet [Kokoro](/kˈOkəɹO/). [[go]]Render it."}
```

- `[word](/phonemes/)` says the word with the IPA phonemes between the slashes, here KO-kuh-roh. `ˈ` marks
  primary stress and `ˌ` secondary; misaki's phoneme list names the symbols it accepts.
- `[word](-1)` lowers the word's stress one step, `[word](-2)` removes it, and `[word](+1)` or `[word](+2)`
  raises it. `+2` is the most: misaki marks one primary stress, so `+3` and higher say it as `+2` does.

The words file, captions and word ids keep the word as written (`Kokoro.`, id `kokoro`), and `[[go]]`
marks work around it. Keep each link to one word: misaki joins `[New York](...)` into a single timed word.

## Engines

Two engines run the same model, Kokoro-82M v1.0, with the same voices, the same pronunciation controls and the
same word times. `KOKORO_ENGINE=torch` or `KOKORO_ENGINE=onnx` picks one; without it, the plugin uses `torch`
when the `kokoro` package is installed and `onnx` otherwise.

- **torch** calls the `kokoro` package (Python 3.10 to 3.12, PyTorch).
- **onnx** runs [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)'s export of the model on onnxruntime.
  The first run downloads `kokoro-v1.0.onnx` (310 MB) and `voices-v1.0.bin` (27 MB, all 54 voices) from that
  project's `model-files-v1.1` release into `~/.cache/kinescript-kokoro` (`KOKORO_ONNX_DIR` moves it;
  `KOKORO_ONNX_MODEL` and `KOKORO_ONNX_VOICES` name files already on disk). It speaks English voices only, and a
  `.pt` file as a speaker needs the torch engine.

On the ten lines of a 2¼-minute explainer, the onnx engine gave every word the same start and end time as the
torch engine, to the sample, and installed in about 840 MB against 1.7 GB. The export is built with Kokoro's
ONNX-friendly STFT, whose output differs from PyTorch's: as exported, the voice is about 3 dB louder below 1 kHz
and 2 dB quieter above it. The plugin replaces three of the export's weight tensors when it loads the model, after
which the onnx and torch engines differ no more than two torch runs do. `KOKORO_ONNX_STFT=export` keeps the
export's weights. The measurements are in
[oaustegard/experiments, kokoro-onnx-kinescript](https://github.com/oaustegard/experiments/tree/main/kokoro-onnx-kinescript).

## Device

It runs on the GPU when the installed PyTorch was built for it, and on the CPU otherwise (a GPU newer than the
PyTorch build, such as an RTX 50 series card with a CUDA 12.4 build, would fail on the first line).
`KOKORO_DEVICE=cpu` or `KOKORO_DEVICE=cuda` picks one. The onnx engine runs on the CPU.

## Not affiliated

This plugin is not affiliated with or endorsed by the makers of Kokoro or kokoro-onnx; it calls the `kokoro`
package or runs kokoro-onnx's export, whose own licences apply to them and to the voices. The onnx engine's
chunking and word-timing functions follow `kokoro/pipeline.py` (Apache-2.0).
