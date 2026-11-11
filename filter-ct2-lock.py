#!/usr/bin/env python3
"""Filter ctranslate2 from the CUDA lock file.

We build CTranslate2 from source with sm_52 support, so we need to remove
the PyPI entry from the lock file before installing the rest of the dependencies.
"""
import re
import sys


def filter_lock(input_path: str, output_path: str) -> None:
    """Remove ctranslate2 entries from a pip lock file."""
    text = open(input_path).read()
    lines = text.split("\n")
    out = []
    skip = False
    for line in lines:
        if re.match(r"^ctranslate2==", line):
            skip = True
            continue
        if skip and line and not line[0].isspace():
            skip = False
        if not skip:
            out.append(line)
    open(output_path, "w").write("\n".join(out))
    print(f"Filtered ctranslate2 from {input_path} -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.lock> <output.lock>")
        sys.exit(1)
    filter_lock(sys.argv[1], sys.argv[2])
