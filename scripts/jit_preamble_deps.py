#!/usr/bin/env python3
"""Drop-in replacement for the `xcrun metal -E -H` include walk in MLX's
make_compiled_preamble.sh, for machines without the Metal offline compiler.

Emits the quote-included headers of a metal kernel header in inclusion order
(post-order DFS, #pragma-once semantics), one path per line, RELATIVE to
SRC_DIR (symlink-safe: both sides are canonicalized before relativizing, so
no caller-side prefix stripping is needed). Angle includes (<metal_stdlib>)
are ignored — the runtime Metal compiler resolves those.

Usage: jit_preamble_deps.py SRC_DIR JIT_INCLUDES_DIR INPUT_FILE
"""
import re
import sys
from pathlib import Path

INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def main(src_dir, jit_dir, input_file):
    src_dir = Path(src_dir).resolve()
    search = [Path(jit_dir).resolve(), src_dir]
    visited = set()
    order = []

    def resolve(name, including_file):
        cands = [including_file.parent / name] + [d / name for d in search]
        for c in cands:
            if c.is_file():
                return c.resolve()
        return None

    def visit(path, is_root=False):
        if path in visited:
            return
        visited.add(path)
        text = path.read_text()
        for name in INCLUDE_RE.findall(text):
            child = resolve(name, path)
            if child is None:
                print(f"warning: unresolved include {name!r} in {path}", file=sys.stderr)
                continue
            visit(child)
        if not is_root:
            order.append(path)

    visit(Path(input_file).resolve(), is_root=True)
    for p in order:
        try:
            print(p.relative_to(src_dir))
        except ValueError:
            print(f"warning: header outside SRC_DIR skipped: {p}", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:4])
