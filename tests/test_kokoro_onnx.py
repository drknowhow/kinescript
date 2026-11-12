"""kinescript-kokoro's onnx engine: the STFT weights it puts in the export, and KPipeline's word-timing rule.

Needs numpy only; the model, onnxruntime and misaki are not loaded.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_path = Path(__file__).resolve().parents[1] / "plugins" / "kinescript-kokoro" / "kinescript_kokoro" / "onnx.py"
_spec = importlib.util.spec_from_file_location("kokoro_onnx_engine", _path)
K = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)  # @dataclass looks its module up there
_spec.loader.exec_module(K)

N, HOP = K.N_FFT, K.HOP


def _hann():
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)


def _istft_reference(spec):
    """torch.istft's interior, written out: irfft each frame, window it, overlap-add, divide by the window sum."""
    frames = np.fft.irfft(spec, n=N, axis=0) * _hann()[:, None]
    out = np.zeros(HOP * (spec.shape[1] - 1) + N)
    norm = np.zeros_like(out)
    for t in range(spec.shape[1]):
        out[t * HOP:t * HOP + N] += frames[:, t]
        norm[t * HOP:t * HOP + N] += _hann() ** 2
    return out / np.where(norm > 1e-8, norm, 1)


def _istft_custom(spec, w):
    """CustomSTFT.inverse with the given rows: real and imaginary parts through a transposed convolution."""
    real, imag = w[K.STFT + "weight_backward_real"][:, 0], w[K.STFT + "weight_backward_imag"][:, 0]
    out = np.zeros(HOP * (spec.shape[1] - 1) + N)
    for t in range(spec.shape[1]):
        out[t * HOP:t * HOP + N] += spec[:, t].real @ real - spec[:, t].imag @ imag
    return out


def test_inverse_rows_match_torch_istft_inside_the_signal():
    rng = np.random.default_rng(0)
    spec = np.exp(rng.normal(0, 0.5, (N // 2 + 1, 60))) * np.exp(1j * rng.uniform(-np.pi, np.pi, (N // 2 + 1, 60)))
    spec[0].imag = spec[-1].imag = 0  # a real signal's DC and Nyquist bins
    got, want = _istft_custom(spec, K.stft_weights()), _istft_reference(spec)
    inside = slice(N, -N)  # torch.istft scales the edges by the partial window sum
    assert np.abs(got[inside] - want[inside]).max() < 1e-5


def test_forward_imaginary_rows_are_zero_at_dc_and_nyquist():
    imag = K.stft_weights()[K.STFT + "weight_forward_imag"][:, 0]
    assert not imag[0].any() and not imag[N // 2].any()  # exact zeros, so the phase there is 0 or pi as in torch
    k, n = np.arange(1, N // 2), np.arange(N)
    assert np.allclose(imag[1:N // 2], -np.sin(2 * np.pi * np.outer(k, n) / N) * _hann(), atol=1e-7)


def test_join_timestamps_follows_kpipeline():
    # "Hi there." -> tokens with their phonemes; durations in frames: pad, h, I, space, ð, ɛ, ɹ, ., pad
    tokens = [SimpleNamespace(phonemes="hI", whitespace=" "), SimpleNamespace(phonemes="ðɛɹ", whitespace=""),
              SimpleNamespace(phonemes=".", whitespace="")]
    K.join_timestamps(tokens, np.array([5, 4, 6, 2, 3, 5, 4, 8, 1]))
    # In half-frames, as kokoro/pipeline.py counts: "hI" starts at 2*(5-3) = 4 and ends at 4 + 2*10 + 2 = 26;
    # "ðɛɹ" starts there and ends at 28 + 2*12 = 52 (the space counts once more); "." runs from 52 to 68
    assert (tokens[0].start_ts, tokens[0].end_ts) == (4 / 80, 26 / 80)
    assert (tokens[1].start_ts, tokens[1].end_ts) == (26 / 80, 52 / 80)
    assert (tokens[2].start_ts, tokens[2].end_ts) == (52 / 80, 68 / 80)


def test_en_tokenize_splits_long_text_at_the_last_stop():
    tok = [SimpleNamespace(phonemes="a" * 99, whitespace=" ") for _ in range(4)]
    tok += [SimpleNamespace(phonemes=".", whitespace=" ")]
    tok += [SimpleNamespace(phonemes="b" * 99, whitespace=" ") for _ in range(4)]
    chunks = list(K.en_tokenize(tok))
    assert len(chunks) == 2 and chunks[0][0].endswith(".") and all(len(ps) <= K.LIMIT for ps, _ in chunks)
