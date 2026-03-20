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
            name="w_o_sim",
            description="Ablation: remove the similarity branch and keep only heterogeneous representations.",
            model_ablation=AblationConfig(disable_similarity_branch=True),
        )
    )
