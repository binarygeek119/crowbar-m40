#!/usr/bin/env python3
"""Guard AWQ kernel bodies so CTranslate2 compiles for sm_52.

CTranslate2's AWQ kernels (4-bit LLM weight quantization) use native FP16
arithmetic that requires compute capability 5.3+. The Tesla M40 is 5.2.
Since faster-whisper/Whisper never uses AWQ quantization, we wrap each
__global__ kernel body with ``#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530``
so the kernels compile (as no-ops for sm_52) and the host-side symbols still
resolve at link time.

The brace counting is aware of string literals and comments, because the
AWQ GEMM kernel contains inline PTX assembly with literal braces.
"""
from __future__ import annotations

import sys


def find_kernel_bodies(text: str) -> list[tuple[int, int]]:
    """Return (open_brace_pos, close_brace_pos) for every __global__ function."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        idx = text.find("__global__", pos)
        if idx == -1:
            break
        brace = text.find("{", idx)
        if brace == -1:
            break
        depth = 1
        j = brace + 1
        n = len(text)
        while j < n and depth > 0:
            c = text[j]
            if c == "/" and j + 1 < n and text[j + 1] == "/":
                nl = text.find("\n", j)
                j = n if nl == -1 else nl
                continue
            if c == "/" and j + 1 < n and text[j + 1] == "*":
                end = text.find("*/", j + 2)
                j = n if end == -1 else end + 2
                continue
            if c == '"':
                j += 1
                while j < n and text[j] != '"':
                    if text[j] == "\\":
                        j += 1
                    j += 1
                j += 1
                continue
            if c == "'":
                j += 1
                while j < n and text[j] != "'":
                    if text[j] == "\\":
                        j += 1
                    j += 1
                j += 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        spans.append((brace, j))
        pos = j
    return spans


def patch(path: str) -> None:
    with open(path) as f:
        text = f.read()
    spans = find_kernel_bodies(text)
    for open_b, close_b in reversed(spans):
        text = text[:close_b - 1] + "\n#endif\n" + text[close_b - 1:]
        text = text[:open_b + 1] + "\n#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530\n" + text[open_b + 1:]
    with open(path, "w") as f:
        f.write(text)
    print(f"patched {path}: {len(spans)} kernel(s) guarded")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        patch(p)
