"""Kokoro on onnxruntime: the same model, voices, misaki pronunciation and word times as KPipeline, without PyTorch.

kokoro (the PyTorch package) pins Python below 3.13; misaki pins it too but runs on 3.13. This engine keeps misaki
for text -> phonemes, so [word](/phonemes/) links, [word](+2) stress and [150](#a&#) number flags work as they do
with KPipeline, and runs Kokoro-82M v1.0 as an ONNX graph that also reports how many frames each phoneme lasts.
Word times come from those durations by the rule KPipeline uses, so they match the torch engine's to the sample.

The export runs kokoro's CustomSTFT (KModel(disable_complex=True)), which differs from the torch.stft and
torch.istft the model was trained with in two places. Its inverse leaves out two weights torch.istft applies: bins
1 to n_fft/2 - 1 of a one-sided spectrum count twice, and the overlap-added squared window (1.5 here) divides the
sum, so as exported the voice is about 3 dB louder below 1 kHz and 2 to 2.5 dB quieter above it. Its forward
transform's Nyquist row of -sin weights is about 4e-15 instead of 0, so that bin's phase, an input to the vocoder,
flips between +pi and -pi where torch says pi. stft_weights() replaces the three weight tensors with ones that
match torch to float precision; KOKORO_ONNX_STFT=export keeps the export's.

The chunking and timing functions below follow kokoro/pipeline.py (hexgrad/kokoro 0.9.4, Apache-2.0) line for
line, so a sound split or timed differently here is a bug.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# thewh1teagle/kokoro-onnx's export of hexgrad/Kokoro-82M v1.0 with a "duration" output and the vocabulary in the
# graph's metadata. The model-files-v1.0 release's file of the same name has neither.
RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
MODEL = "kokoro-v1.0.onnx"
VOICES = "voices-v1.0.bin"
LIMIT = 510  # phonemes per model call, as in KPipeline
STFT = "kmodel.decoder.generator.stft."  # CustomSTFT's buffers, as the export names its initializers
N_FFT, HOP = 20, 5  # Kokoro-82M's istftnet gen_istft_n_fft and gen_istft_hop_size
_loaded: dict = {}  # one session and one voices file per process, shared by the a and b pipelines


def _home() -> Path:
    if os.environ.get("KOKORO_ONNX_DIR"):
        return Path(os.environ["KOKORO_ONNX_DIR"])
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "kinescript-kokoro"


def _file(env: str, name: str) -> str:
    """The path in env if set, else name in the cache folder, downloaded on first use."""
    if os.environ.get(env):
        return os.environ[env]
    path = _home() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp, \
                urllib.request.urlopen(RELEASE + name) as r:
            shutil.copyfileobj(r, tmp)
        os.replace(tmp.name, path)
    return str(path)


def model_path() -> str:
    """KOKORO_ONNX_MODEL, else kokoro-v1.0.onnx (310 MB) in the cache folder, downloaded on first use."""
    return _file("KOKORO_ONNX_MODEL", MODEL)


def voices_path() -> str:
    """KOKORO_ONNX_VOICES, else voices-v1.0.bin (27 MB, all 54 voices) in the cache folder."""
    return _file("KOKORO_ONNX_VOICES", VOICES)


def stft_weights(n_fft: int = N_FFT, hop: int = HOP) -> dict[str, np.ndarray]:
    """CustomSTFT weights ([bins, 1, n_fft]) that make it torch.stft and torch.istft: the forward -sin rows with exact
    zeros at DC and Nyquist, and the inverse rows counting every bin but DC and Nyquist twice, divided by the
    overlap-added squared periodic Hann window."""
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    k, n = np.arange(n_fft // 2 + 1), np.arange(n_fft)
    angle = 2 * np.pi * np.outer(k, n) / n_fft
    edge = (k == 0) | (k == n_fft // 2)
    forward_imag = np.where(edge[:, None], 0.0, -np.sin(angle)) * window
    rows = window / n_fft * (np.where(edge, 1.0, 2.0) / ((window ** 2).sum() / hop))[:, None]

    def f32(a):
        return np.ascontiguousarray(a[:, None, :], dtype=np.float32)
    return {STFT + "weight_forward_imag": f32(forward_imag),
            STFT + "weight_backward_real": f32(np.cos(angle) * rows),
            STFT + "weight_backward_imag": f32(np.sin(angle) * rows)}


def session():
    """The model's onnxruntime session, its STFT weights replaced (stft_weights) unless KOKORO_ONNX_STFT=export."""
    if "session" not in _loaded:
        import onnxruntime as rt
        options = rt.SessionOptions()
        if os.environ.get("KOKORO_ONNX_STFT") != "export":
            # An OrtValue given to add_initializer must outlive the session, hence _loaded
            _loaded["stft"] = {k: rt.OrtValue.ortvalue_from_numpy(v) for k, v in stft_weights().items()}
            for name, value in _loaded["stft"].items():
                options.add_initializer(name, value)
        sess = rt.InferenceSession(model_path(), options, providers=["CPUExecutionProvider"])
        meta = sess.get_modelmeta().custom_metadata_map.get("kokoro_config")
        if not meta or "duration" not in {o.name for o in sess.get_outputs()}:
            raise ValueError(f"{model_path()} has no duration output or no vocabulary; use {MODEL} from "
                             f"{RELEASE}")
        config = json.loads(meta)
        if (config["istftnet"]["gen_istft_n_fft"], config["istftnet"]["gen_istft_hop_size"]) != (N_FFT, HOP):
            raise ValueError(f"{model_path()} is not Kokoro-82M v1.0: its istftnet differs")
        _loaded["session"], _loaded["vocab"] = sess, config["vocab"]
    return _loaded["session"]


def voices():
    if "voices" not in _loaded:
        _loaded["voices"] = np.load(voices_path())
    return _loaded["voices"]


@dataclass
class Result:
    audio: np.ndarray
    tokens: list


class Pipeline:
    """KPipeline's English half on onnxruntime: pipe(text, voice=, speed=) yields Results with .audio and .tokens."""

    def __init__(self, lang: str):
        from misaki import en, espeak
        if lang not in ("a", "b"):
            raise ValueError(f"the onnx engine speaks English voices only (a or b), not {lang!r}")
        self.sess = session()
        self.vocab = _loaded["vocab"]
        self._blends: dict = {}
        try:
            fallback = espeak.EspeakFallback(british=lang == "b")
        except Exception:  # KPipeline also goes on without it: unknown words are then skipped
            fallback = None
        self.g2p = en.G2P(trf=False, british=lang == "b", fallback=fallback, unk="")

    def voice(self, name: str) -> np.ndarray:
        """A voice's style pack, [510, 1, 256]; "a,b" averages them, as KPipeline.load_voice does."""
        if name not in self._blends:
            missing = [v for v in name.split(",") if v not in voices()]
            if missing:
                raise ValueError(f"no voice {missing[0]!r} in {voices_path()} (a .pt file needs the torch engine)")
            packs = [voices()[v] for v in name.split(",")]
            self._blends[name] = packs[0] if len(packs) == 1 else np.mean(np.stack(packs), axis=0)
        return self._blends[name]

    def infer(self, ps: str, pack: np.ndarray, speed: float):
        ids = [self.vocab[p] for p in ps if p in self.vocab]  # KModel drops phonemes it has no id for
        audio, dur = self.sess.run(None, {
            "input_ids": np.array([[0, *ids, 0]], dtype=np.int64),
            "style": np.asarray(pack[len(ps) - 1], dtype=np.float32).reshape(1, -1),  # by phoneme count, as KPipeline
            "speed": np.array([speed], dtype=np.float32)})
        return np.asarray(audio, dtype=np.float32).ravel(), np.asarray(dur).ravel()

    def __call__(self, text: str, voice: str, speed: float = 1):
        pack = self.voice(voice)
        for graphemes in re.split(r"\n+", text.strip()):
            if not graphemes.strip():
                continue
            _, tokens = self.g2p(graphemes)
            for ps, tks in en_tokenize(tokens):
                if not ps:
                    continue
                ps = ps[:LIMIT]
                audio, dur = self.infer(ps, pack, speed)
                join_timestamps(tks, dur)
                yield Result(audio=audio, tokens=tks)


def tokens_to_ps(tokens) -> str:
    return "".join(t.phonemes + (" " if t.whitespace else "") for t in tokens).strip()


def waterfall_last(tokens, next_count: int, waterfall=("!.?…", ":;", ",—"), bumps=(")", "”")) -> int:
    for w in waterfall:
        z = next((i for i, t in reversed(list(enumerate(tokens))) if t.phonemes in set(w)), None)
        if z is None:
            continue
        z += 1
        if z < len(tokens) and tokens[z].phonemes in bumps:
            z += 1
        if next_count - len(tokens_to_ps(tokens[:z])) <= LIMIT:
            return z
    return len(tokens)


def en_tokenize(tokens):
    """misaki's tokens -> (phonemes, tokens) chunks of at most 510 phonemes, split at the latest stop."""
    tks, pcount = [], 0
    for t in tokens:
        t.phonemes = "" if t.phonemes is None else t.phonemes
        next_ps = t.phonemes + (" " if t.whitespace else "")
        next_pcount = pcount + len(next_ps.rstrip())
        if next_pcount > LIMIT:
            z = waterfall_last(tks, next_pcount)
            yield tokens_to_ps(tks[:z]), tks[:z]
            tks = tks[z:]
            pcount = len(tokens_to_ps(tks))
            if not tks:
                next_ps = next_ps.lstrip()
        tks.append(t)
        pcount += len(next_ps)
    if tks:
        yield tokens_to_ps(tks).strip(), tks


def join_timestamps(tokens, pred_dur) -> None:
    """Set each token's start_ts and end_ts in seconds from the model's frames per phoneme (40 frames a second),
    splitting each space between the words either side of it."""
    divisor = 80  # half-frames per second
    if not tokens or len(pred_dur) < 3:
        return
    left = right = 2 * max(0, int(pred_dur[0]) - 3)
    i = 1
    for t in tokens:
        if i >= len(pred_dur) - 1:
            break
        if not t.phonemes:
            if t.whitespace:
                i += 1
                left = right + int(pred_dur[i])
                right = left + int(pred_dur[i])
                i += 1
            continue
        j = i + len(t.phonemes)
        if j >= len(pred_dur):
            break
        t.start_ts = left / divisor
        token_dur = int(pred_dur[i:j].sum())
        space_dur = int(pred_dur[j]) if t.whitespace else 0
        left = right + (2 * token_dur) + space_dur
        t.end_ts = left / divisor
        right = left + space_dur
        i = j + (1 if t.whitespace else 0)
