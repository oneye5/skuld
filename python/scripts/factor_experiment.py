from __future__ import annotations

import sys
from pathlib import Path

python_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(python_root / "src"))

from skuld_research.experiments.factor_experiment import main  # noqa: E402,I001


if __name__ == "__main__":
    raise SystemExit(main())
