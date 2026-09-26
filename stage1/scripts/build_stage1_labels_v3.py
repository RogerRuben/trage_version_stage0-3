"""Compatibility wrapper for ``python -m stage1.v3.cli``."""

from stage1.v3.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
