"""Pre-submission check: README numbers match the saved model, and no draft text or placeholders are left."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
readme = (ROOT / "README.md").read_text(encoding="utf-8")
meta = json.loads((ROOT / "models" / "metadata.json").read_text())
test = meta["test_at_selected_threshold"]
problems = []

expected = {
    "threshold": f"{meta['threshold']:.2f}",
    "test PR-AUC": f"{test['PR-AUC']:.4f}",
    "test ROC-AUC": f"{test['ROC-AUC']:.4f}",
}
for name, text in expected.items():
    if text not in readme:
        problems.append(f"README is missing the saved {name}: {text}")

policy = (ROOT / "reports" / "policy_comparison_test.csv").read_text().splitlines()[1].split(",")
contacted, cost = int(policy[1]), float(policy[2])
for name, text in {"customers contacted": f"{contacted:,}", "expected cost": f"${cost:,.0f}"}.items():
    if text not in readme:
        problems.append(f"README is missing the saved {name}: {text}")

cal = {line.split(",")[0]: float(line.split(",")[1]) for line in
       (ROOT / "reports" / "calibration_comparison.csv").read_text().splitlines()[1:]}
for method, brier in cal.items():
    if f"{brier:.4f}" not in readme:
        problems.append(f"README is missing the saved Brier score for {method}: {brier:.4f}")

for stale in ["has not been generated yet", "has not been performed", "<your-username>", "Demo video checklist", "gradio"]:
    if stale.lower() in readme.lower():
        problems.append(f"Draft/outdated text still in README: {stale!r}")
for token in re.findall(r"ADD_[A-Z_]+_LINK", readme):
    problems.append(f"Placeholder not replaced: {token}")
for shot in ["single_prediction.png", "csv_upload.png"]:
    if not (ROOT / "screenshots" / shot).exists():
        problems.append(f"Missing screenshot: screenshots/{shot}")

print("\n".join(["PROBLEMS:"] + [f" - {p}" for p in problems]) if problems else "README checks passed.")
sys.exit(1 if problems else 0)
