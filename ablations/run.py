from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS_DIR = ROOT / "ablations"
DEFAULT_VARIANTS = [
    "w_o_sim",
    "w_o_horv_plus_sim",
    "w_o_lorv_plus_sim",
    "w_o_horv",
    "w_o_lorv",
]


def load_summary(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one or more ablation experiments from a single entry point."
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=DEFAULT_VARIANTS,
        help="Ablation variants to run. Default: run the ablation suite only.",
    )
    parser.add_argument(
        "--base-out-dir",
        type=str,
        default="artifacts/ablations",
        help="Base output directory. Each variant writes to a dedicated subdirectory under it.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip variants whose summary.json already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running the remaining variants when one variant fails.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available variants and exit.",
    )
    args, passthrough = parser.parse_known_args()

    if args.list:
        print("\n".join(DEFAULT_VARIANTS))
        return

    requested = list(dict.fromkeys(args.variants))
    unknown = [name for name in requested if name not in DEFAULT_VARIANTS]
    if unknown:
        raise ValueError("Unknown ablation variants: " + ", ".join(unknown))

    if "--out-dir" in passthrough:
        raise ValueError("Do not pass --out-dir to ablations/run.py; use --base-out-dir instead.")

    base_out_dir = Path(args.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    suite_results: List[Dict[str, object]] = []
    for variant in requested:
        script_path = ABLATIONS_DIR / variant / "train.py"
        out_dir = base_out_dir / variant
        summary_path = out_dir / "summary.json"

        if args.skip_existing and summary_path.exists():
            print(f"[skip] {variant} -> {summary_path}")
            suite_results.append(
                {
                    "name": variant,
                    "status": "skipped",
                    "out_dir": str(out_dir),
                    "summary_path": str(summary_path),
                    "summary": load_summary(summary_path),
                }
            )
            continue

        cmd = [sys.executable, str(script_path), *passthrough, "--out-dir", str(out_dir)]
        print(f"[run] {variant}")
        completed = subprocess.run(cmd, cwd=str(ROOT), check=False)

        record: Dict[str, object] = {
            "name": variant,
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": int(completed.returncode),
            "out_dir": str(out_dir),
            "summary_path": str(summary_path),
        }
        if summary_path.exists():
            record["summary"] = load_summary(summary_path)
        suite_results.append(record)

        if completed.returncode != 0:
            if not args.continue_on_error:
                suite_summary = {"variants": suite_results}
                with open(base_out_dir / "suite_summary.json", "w", encoding="utf-8") as f:
                    json.dump(suite_summary, f, ensure_ascii=False, indent=2)
                raise SystemExit(completed.returncode)
            print(f"[warn] {variant} failed with return code {completed.returncode}")

    suite_summary = {"variants": suite_results}
    with open(base_out_dir / "suite_summary.json", "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, ensure_ascii=False, indent=2)
    print(f"[done] suite summary saved to {base_out_dir / 'suite_summary.json'}")


if __name__ == "__main__":
    main()
