import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
sys.path.insert(0, os.path.join(_ROOT, 'core'))
os.chdir(_ROOT)

import subprocess
import re
import random

ALL_TICKERS = [
    "AAPL", "GOOGL", "MSFT", "NVDA", "META",
    "AMZN", "TSLA", "AMD", "INTC", "NFLX",
    "QCOM", "AVGO", "TXN", "MU", "AMAT",
    "CRM", "ORCL", "CSCO", "IBM", "SNOW",
    "JPM", "BAC", "GS", "V", "MA",
    "BLK", "MS", "C",
    "JNJ", "PFE", "ABBV", "MRK", "UNH", "LLY",
    "COST", "WMT", "HD", "NKE", "MCD", "SBUX",
    "XOM", "CVX",
    "BA", "CAT", "DIS", "PYPL", "UBER", "ADBE", "NOW", "PANW", "TMO",
]

THRESHOLDS  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
N_SAMPLES   = 5
SAMPLE_SIZE = 10

random.seed(42)
ticker_samples = [
    ",".join(random.sample(ALL_TICKERS, SAMPLE_SIZE))
    for _ in range(N_SAMPLES)
]

results = {t: [] for t in THRESHOLDS}
total_runs = len(THRESHOLDS) * N_SAMPLES
run = 0

for thresh in THRESHOLDS:
    print(f"\n{'='*60}")
    print(f"THRESHOLD: {thresh}")
    print("="*60)

    for i, tickers_str in enumerate(ticker_samples):
        run += 1
        print(f"\n  [{run}/{total_runs}] Sample {i+1}: {tickers_str}")

        env = {
            **os.environ,
            "ABLATION_TICKERS":    tickers_str,
            "ABLATION_THRESHOLD":  str(thresh),
        }

        subprocess.run(
            [sys.executable, os.path.join(_ROOT, "training", "ml_optimizer.py")],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        out = subprocess.run(
            [sys.executable, os.path.join(_ROOT, "simulation", "simulation.py")],
            env=env, capture_output=True, text=True
        )

        roi_match = re.search(r'ROI:\s*\+?([-\d.]+)%', out.stdout)
        roi = float(roi_match.group(1)) if roi_match else None
        results[thresh].append(roi)
        print(f"  → ROI: {roi:+.1f}%" if roi is not None else "  → ROI: N/A")

print("\n\n" + "="*60)
print("THRESHOLD ABLATION RESULTS (avg of 5 random ticker sets)")
print("="*60)
print(f"  {'Threshold':<12} {'Avg':>8}  {'Min':>8}  {'Max':>8}")
print(f"  {'-'*44}")
for thresh, rois in results.items():
    valid = [r for r in rois if r is not None]
    avg = sum(valid) / len(valid) if valid else 0
    mn  = min(valid) if valid else 0
    mx  = max(valid) if valid else 0
    bar = "█" * max(0, int(avg / 5))
    print(f"  {thresh:<12} {avg:>+7.1f}%  {mn:>+7.1f}%  {mx:>+7.1f}%  {bar}")
print("="*60)
