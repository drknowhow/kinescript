"""Buffer views compressed with EXT_meshopt_compression (or KHR_meshopt_compression, the same format), as
gltfpack -c writes them, decoded with numpy: vertex attributes (codec versions 0 and 1), triangle indices, index
sequences, and the octahedral, quaternion, exponential and colour filters. It follows meshoptimizer's reference
decoder (meshopt_decoder_reference.js); the byte groups are read in Python, the deltas added up in numpy."""
from __future__ import annotations

import numpy as np

NAMES = ("EXT_meshopt_compression", "KHR_meshopt_compression")
# A byte of packed 1-, 2- or 4-bit deltas, as the values in it (1-bit from the low bit up, the others high first).
_UNPACK = {1: [tuple((b >> s) & 1 for s in range(8)) for b in range(256)],
           2: [tuple((b >> s) & 3 for s in (6, 4, 2)) + (b & 3,) for b in range(256)],
           4: [(b >> 4, b & 15) for b in range(256)]}
_MODES = ((0, 2, 4, 8), (0, 1, 2, 4), (1, 2, 4, 8))  # bits per delta: version 0, version 1 by control 0 and 1
_DEZIG8 = np.array([(v >> 1) ^ (-(v & 1) & 0xFF) for v in range(256)], np.uint8)


def decode(ext: dict, data: bytes) -> bytes:
    """The bytes of the buffer view that `ext` (its meshopt extension object) packs into `data` (its buffer)."""
    count, stride, mode = int(ext["count"]), int(ext["byteStride"]), ext.get("mode")
    off = int(ext.get("byteOffset", 0))
    src = bytes(data[off:off + int(ext["byteLength"])])
    try:
        if mode == "ATTRIBUTES":
            out = _filter(_vertices(src, count, stride), stride, ext.get("filter", "NONE"))
        elif mode == "TRIANGLES":
            out = _triangles(src, count, stride)
        elif mode == "INDICES":
            out = _sequence(src, count, stride)
        else:
            raise ValueError(f"meshopt mode {mode!r} is not ATTRIBUTES, TRIANGLES or INDICES")
    except IndexError:
        raise ValueError(f"meshopt {mode} data ends early") from None
    return out.tobytes()


def _vertices(src: bytes, count: int, stride: int) -> np.ndarray:
    """count elements of stride bytes: per block of elements, each byte's deltas from the element before (byte
    groups of 16, packed by a 2-bit mode each), then added up."""
    if not src or src[0] not in (0xA0, 0xA1):
        raise ValueError(f"meshopt vertex data starts {src[:1].hex() or 'empty'}, not a0 or a1")
    if stride % 4 or not 0 < stride <= 256:
        raise ValueError(f"meshopt vertex stride {stride} is not a multiple of 4 up to 256")
    version = src[0] & 15
    block = min((0x2000 // stride) & ~15, 256)
    tail = stride if version == 0 else stride + stride // 4
    base = len(src) - tail
    if base < 1:
        raise IndexError
    last = np.frombuffer(src, np.uint8, stride, base).copy()  # what the first element's deltas are from
    chan = src[base + stride:base + tail]  # version 1: how each 4 bytes are deltas (bytes, 16-bit or xor 32-bit)
    out = np.empty((count, stride), np.uint8)
    pos = 1
    for start in range(0, count, block):
        n = min(count - start, block)
        groups = (n + 15) >> 4
        heads = (groups + 3) >> 2
        control = pos
        pos += stride // 4 if version else 0
        deltas = np.zeros((stride, groups * 16), np.uint8)
        for k in range(stride):
            cm = (src[control + (k >> 2)] >> ((k & 3) << 1)) & 3 if version else 0
            if cm == 2:  # all zero
                continue
            if cm == 3:  # stored as they are
                deltas[k, :n] = np.frombuffer(src, np.uint8, n, pos)
                pos += n
                continue
            head, pos = pos, pos + heads
            modes = _MODES[cm + 1 if version else 0]
            row = bytearray(groups * 16)
            for g in range(groups):
                bits = modes[(src[head + (g >> 2)] >> ((g & 3) << 1)) & 3]
                if bits == 8:
                    row[g * 16:g * 16 + 16] = src[pos:pos + 16]
                    pos += 16
                elif bits:
                    table, top = _UNPACK[bits], (1 << bits) - 1
                    vals = [v for b in src[pos:pos + 2 * bits] for v in table[b]]
                    pos += 2 * bits
                    if top in vals:  # a full value says the delta is the next byte after the group
                        for m, v in enumerate(vals):
                            if v == top:
                                vals[m] = src[pos]
                                pos += 1
                    row[g * 16:g * 16 + 16] = bytes(vals)
            deltas[k] = np.frombuffer(bytes(row), np.uint8)
        d = deltas[:, :n]
        rows = out[start:start + n]
        for b in range(0, stride, 4):
            cm = chan[b >> 2] & 3 if version else 0
            if cm == 0:  # each byte: zigzag deltas, added up
                rows[:, b:b + 4] = (np.cumsum(_DEZIG8[d[b:b + 4]], 1, dtype=np.uint8) + last[b:b + 4, None]).T
            elif cm == 1:  # 16-bit zigzag deltas
                for h in (b, b + 2):
                    v = d[h].astype(np.uint16) | d[h + 1].astype(np.uint16) << 8
                    v = (v >> 1) ^ (np.uint16(0) - (v & 1))
                    v = np.cumsum(v, dtype=np.uint16) + np.uint16(int(last[h]) | int(last[h + 1]) << 8)
                    rows[:, h], rows[:, h + 1] = v & 0xFF, v >> 8
            elif cm == 2:  # 32-bit values xor the one before, rotated
                v = (d[b].astype(np.uint32) | d[b + 1].astype(np.uint32) << 8 | d[b + 2].astype(np.uint32) << 16
                     | d[b + 3].astype(np.uint32) << 24)
                rot = chan[b >> 2] >> 4
                if rot:
                    v = (v >> np.uint32(rot)) | (v << np.uint32(32 - rot))
                v = np.bitwise_xor.accumulate(v) ^ np.frombuffer(last[b:b + 4].tobytes(), "<u4")[0]
                rows[:, b:b + 4] = v.astype("<u4").view(np.uint8).reshape(n, 4)
            else:
                raise ValueError("meshopt vertex channel mode 3 is not defined")
        last = rows[-1].copy()
    if pos != len(src) - max(tail, 32 if version == 0 else 24):
        raise ValueError("meshopt vertex data does not end where its blocks do")
    return out


def _filter(raw: np.ndarray, stride: int, how: str) -> np.ndarray:
    """Decoded elements with their filter undone: unit vectors from octahedral x and y, quaternions from their
    three smallest components, floats from a shared exponent, colours from YCoCg."""
    if how in (None, "NONE"):
        return raw
    if how == "OCTAHEDRAL":
        kind, top = {4: ("<i1", 127), 8: ("<i2", 32767)}[stride]
        q = raw.view(kind).astype(np.float64)
        x, y = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
        z = 1.0 - np.abs(x) - np.abs(y)
        t = np.maximum(-z, 0.0)
        x, y = x - np.where(x >= 0, t, -t), y - np.where(y >= 0, t, -t)
        h = top / np.sqrt(x * x + y * y + z * z)
        q[:, 0], q[:, 1], q[:, 2] = np.floor(x * h + 0.5), np.floor(y * h + 0.5), np.floor(z * h + 0.5)
        return q.astype(kind).view(np.uint8)
    if how == "QUATERNION":
        if stride != 8:
            raise ValueError(f"meshopt quaternion filter on stride {stride}, not 8")
        q = raw.view("<i2").astype(np.int64)
        big, s = q[:, 3] & 3, np.sqrt(0.5) / (q[:, 3] | 3)
        x, y, z = q[:, 0] * s, q[:, 1] * s, q[:, 2] * s
        w = np.sqrt(np.maximum(0.0, 1.0 - x * x - y * y - z * z))
        out = np.empty((len(q), 4), np.int64)
        rows = np.arange(len(q))
        for k, v in ((1, x), (2, y), (3, z), (0, w)):
            out[rows, (big + k) % 4] = np.floor(v * 32767 + 0.5)
        return out.astype("<i2").view(np.uint8)
    if how == "EXPONENTIAL":
        v = raw.view("<i4").astype(np.int64)
        m = ((v & 0xFFFFFF) ^ 0x800000) - 0x800000
        return np.ldexp(m.astype(np.float64), (v >> 24).astype(np.int32)).astype("<f4").view(np.uint8)
    if how == "COLOR":
        kind, top = {4: (("<u1", "<i1"), 255), 8: (("<u2", "<i2"), 65535)}[stride]
        u, s = raw.view(kind[0]).astype(np.int64), raw.view(kind[1]).astype(np.int64)
        yy, co, cg, a_in = u[:, 0], s[:, 1], s[:, 2], u[:, 3]
        bit = np.floor(np.log2(np.maximum(a_in, 1))).astype(np.int64) * (a_in > 0) - (a_in == 0)
        scale = (np.int64(1) << (bit + 1)) - 1
        a = a_in & (scale >> 1)
        a = (a << 1) | (a & 1)
        ss = top / np.where(scale > 0, scale, 1)
        out = np.stack([yy + co - cg, yy + cg, yy - co - cg, a], 1) * ss[:, None]
        return np.floor(out + 0.5).astype(np.int64).astype(kind[0]).view(np.uint8)
    raise ValueError(f"meshopt filter {how!r} is not NONE, OCTAHEDRAL, QUATERNION, EXPONENTIAL or COLOR")


def _leb(src: bytes, pos: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        b = src[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        if b < 0x80:
            return n & 0xFFFFFFFF, pos
        shift += 7


def _zig(v: int) -> int:
    return (v >> 1) ^ -(v & 1)


def _triangles(src: bytes, count: int, size: int) -> np.ndarray:
    """Triangle indices, each triangle a code byte: an edge from the last 16 and a vertex from the last 16, the
    next new one, or one given as a delta (codec version 1)."""
    if not src or src[0] != 0xE1:
        raise ValueError(f"meshopt triangle data starts {src[:1].hex() or 'empty'}, not e1")
    if count % 3:
        raise ValueError(f"meshopt triangle data holds {count} indices, not a multiple of 3")
    tris = count // 3
    code, pos, aux = 1, 1 + tris, len(src) - 16
    edge, eo, vert, vo = [0] * 32, 0, [0] * 16, 0
    nxt = last = 0
    out = []
    for _ in range(tris):
        c0 = src[code]
        code += 1
        b0, b1 = c0 >> 4, c0 & 15
        if b0 < 15:
            a, b = edge[(eo - 1 - 2 * b0) & 31], edge[(eo - 2 - 2 * b0) & 31]
            if b1 == 0:
                c, nxt = nxt, nxt + 1
            elif b1 < 13:
                c = vert[(vo - 1 - b1) & 15]
            elif b1 == 13:
                last -= 1
                c = last
            elif b1 == 14:
                last += 1
                c = last
            else:
                v, pos = _leb(src, pos)
                last += _zig(v)
                c = last
            if b1 == 0 or b1 >= 13:
                vert[vo], vo = c, (vo + 1) & 15
            for e in (b, c, c, a):  # pushed backwards
                edge[eo], eo = e, (eo + 1) & 31
        else:
            if b1 < 14:
                e = src[aux + b1]
                z, w = e >> 4, e & 15
                a, nxt = nxt, nxt + 1
                if z == 0:
                    b, nxt = nxt, nxt + 1
                else:
                    b = vert[(vo - z) & 15]
                if w == 0:
                    c, nxt = nxt, nxt + 1
                else:
                    c = vert[(vo - w) & 15]
                fresh = (a, *((b,) if z == 0 else ()), *((c,) if w == 0 else ()))
            else:
                e = src[pos]
                pos += 1
                if e == 0:
                    nxt = 0
                z, w = e >> 4, e & 15
                if b1 == 14:
                    a, nxt = nxt, nxt + 1
                else:
                    v, pos = _leb(src, pos)
                    last += _zig(v)
                    a = last
                if z == 0:
                    b, nxt = nxt, nxt + 1
                elif z == 15:
                    v, pos = _leb(src, pos)
                    last += _zig(v)
                    b = last
                else:
                    b = vert[(vo - z) & 15]
                if w == 0:
                    c, nxt = nxt, nxt + 1
                elif w == 15:
                    v, pos = _leb(src, pos)
                    last += _zig(v)
                    c = last
                else:
                    c = vert[(vo - w) & 15]
                fresh = (a, *((b,) if z in (0, 15) else ()), *((c,) if w in (0, 15) else ()))
            for f in fresh:
                vert[vo], vo = f, (vo + 1) & 15
            for e in (a, b, b, c, c, a):
                edge[eo], eo = e, (eo + 1) & 31
        out += (a, b, c)
    if pos > aux:
        raise IndexError
    return np.array(out, np.int64).astype({2: "<u2", 4: "<u4"}[size])


def _sequence(src: bytes, count: int, size: int) -> np.ndarray:
    """Indices, each a zigzag delta from one of two earlier ones (codec version 1)."""
    if not src or src[0] != 0xD1:
        raise ValueError(f"meshopt index data starts {src[:1].hex() or 'empty'}, not d1")
    last, pos, out = [0, 0], 1, []
    for _ in range(count):
        v, pos = _leb(src, pos)
        k = v & 1
        last[k] = (last[k] + _zig(v >> 1)) & 0xFFFFFFFF
        out.append(last[k])
    return np.array(out, np.int64).astype({2: "<u2", 4: "<u4"}[size])
