"""Jalankan: python -m src"""

from __future__ import annotations

import sys

from .app import main as run_app


def main() -> int:
    return run_app()


if __name__ == "__main__":
    sys.exit(main())
