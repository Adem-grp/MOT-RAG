"""Step 1a: download a starter set of MOT papers from arXiv into data/papers."""
import time
from pathlib import Path

import requests

# arXiv IDs -> readable names. Add your own later.
PAPERS = {
    "1602.00763": "SORT",
    "1703.07402": "DeepSORT",
    "1903.05625": "Tracktor",
    "2004.01177": "CenterTrack",
    "2004.01888": "FairMOT",
    "2012.15460": "TransTrack",
    "2101.02702": "TrackFormer",
    "2105.03247": "MOTR",
    "2110.06864": "ByteTrack",
    "2202.13514": "StrongSORT",
    "2203.14360": "OC-SORT",
    "2206.14651": "BoT-SORT",
}

OUT_DIR = Path("data/papers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for arxiv_id, name in PAPERS.items():
    target = OUT_DIR / f"{name}.pdf"
    if target.exists():
        print(f"skip {name} (already downloaded)")
        continue
    print(f"downloading {name} ({arxiv_id})")
    resp = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=60)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    time.sleep(3)  # be polite to arXiv

print("done")