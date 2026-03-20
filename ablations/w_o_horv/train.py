from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablations.model_variants import AblationConfig
from ablations.shared import AblationSpec, run_ablation


if __name__ == "__main__":
    run_ablation(
        AblationSpec(
            name="w_o_horv",
            description="Ablation: remove the high-order relational view while keeping similarity and low-order views.",
            model_ablation=AblationConfig(disable_high_order_relations=True),
        )
    )
