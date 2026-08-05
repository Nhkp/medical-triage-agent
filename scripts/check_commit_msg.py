#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERN = re.compile(r"^(feat|fix|docs|test|refactor|chore|ci|build)(\([a-z0-9_.-]+\))?: .{1,72}$")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_commit_msg.py <commit-msg-file>", file=sys.stderr)
        return 2
    first_line = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()[0]
    if PATTERN.match(first_line):
        return 0
    print(
        "commit message must match: type(optional-scope): short description",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
