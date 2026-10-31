# kinescript-chatterbox

[Chatterbox](https://github.com/resemble-ai/chatterbox) voices for `kine voice`, from Resemble AI's open models. It
speaks the `text` of a scene's sounds into WAV files on your own machine and writes the time of each word, so
layers can start on a word and captions follow the voice. Chatterbox Turbo takes tags such as `[laugh]` and
`[sigh]`, though it acts them only some of the time (see Tags), and either model can copy the voice of a short clip.

```json
{"id": "vo1", "role": "voice", "backend": "chatterbox",
 "text": "Well [sigh] that render took a while. [[go]][laugh] But look at it."}
```

```bash
kine voice scene.json     # writes voice/vo1.wav and voice/vo1.words.json
kine check scene.json
```

## Install

It needs Kinescript 1.3 or newer, and Chatterbox needs PyTorch. Into the same environment as Kinescript:

```bash
pip install kinescript-chatterbox
```

From a checkout of [Kinescript](https://github.com/drknowhow/kinescript), `pip install ./plugins/kinescript-chatterbox`
does the same.

On Python 3.10 to 3.13, Chatterbox pins PyTorch 2.6, whose CUDA builds stop before the RTX 50 series, so on such a
card it runs on the CPU. On Python 3.14 it takes PyTorch 2.9 or newer; install a CUDA build first to use the GPU:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install kinescript-chatterbox
```

To keep it out of Kinescript's environment, install it into another Python and point `KINE_VOICE_CMD_CHATTERBOX`
at it; `kine voice` sends it the lines as JSON on stdin and never through a shell:

```bash
python -m venv ~/chatterbox-env
~/chatterbox-env/bin/pip install kinescript-chatterbox
export KINE_VOICE_CMD_CHATTERBOX='["'$HOME'/chatterbox-env/bin/python", "-m", "kinescript_chatterbox"]'
```

The first run downloads Chatterbox Turbo (4.0 GB) and wav2vec2 (380 MB) from Hugging Face. The standard model is a
separate download, made the first time a sound asks for it.

## Tags

Turbo (in chatterbox-tts 0.1.7) knows 19 tags:

- sounds: `[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[groan]`, `[cough]`, `[clear throat]`, `[sniff]`, `[shush]`
- delivery: `[whispering]`, `[crying]`, `[angry]`, `[happy]`, `[sarcastic]`, `[surprised]`, `[fear]`, `[dramatic]`,
  `[narration]`, `[advertisement]`

A tag written the way ElevenLabs writes it, such as `[laughs]`, `[whispers]` or `[clears throat]`, is read as
Turbo's `[laugh]`, `[whispering]` or `[clear throat]`, so one scene works with both. A tag Turbo does not know is left
out, so it is not read aloud, and listed under `ignored` in what `kine voice` prints.

Turbo acts its tags only some of the time. With the built-in voice and seeds 0 to 2, nine takes with `[laugh]`
(three lines) had no sound where the laugh goes in three: a silent pause in two, no pause in one. The other six had
sound there, not checked by ear. Seven takes with `[whispering]` never whispered: the words stayed 81 to 99% voiced,
like plain speech (86 to 96%), where ElevenLabs' whisper is 1% voiced. Listen to a tagged line before you rely on
it, try another `seed` in `options`, or give that sound `"backend": "elevenlabs"`.

## Voices and settings

`speaker` is a WAV clip of 5 s or more, relative to the scene file, whose voice Chatterbox copies. Without one it
speaks in the model's own voice. `rate` is not supported; Chatterbox speaks at one speed, and `kine voice` lists a
rate other than 1 under `ignored`.

`options`:

- `model`: `turbo` (the default) acts on tags. `standard` is the larger original model, which has no tags but takes
  `exaggeration` (0.5 by default; higher is more emotional) and `cfg_weight` (lower is slower and more deliberate).
- `temperature`, `top_p`, `top_k` (Turbo), `min_p` (standard), `repetition_penalty`: sampling settings.
- `seed`: 0 by default. The same text, voice and seed speak the same way again on the same machine.

Chatterbox puts Resemble AI's inaudible Perth watermark in everything it speaks.

## Word times

Chatterbox gives no word times, so the plugin finds them: wav2vec2 (`facebook/wav2vec2-base-960h`) hears the letters
in each 20 ms of the sound, and a forced alignment fits the text's letters to them. Numbers are aligned as they are
read ("250" as two hundred fifty, "2026" as twenty twenty six, "21st" as twenty first). A tag is matched to whatever
sound fills its place, so a laugh does not stretch the words around it. Word times come for English only.

## Device

It runs on the GPU when the installed PyTorch was built for it, and on the CPU otherwise. `CHATTERBOX_DEVICE=cpu`
or `CHATTERBOX_DEVICE=cuda` picks one.
