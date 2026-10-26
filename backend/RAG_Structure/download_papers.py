"""Download the MOT research references into data/papers.

The original starter set is kept in ``PAPERS``.  ``ADDITIONAL_PAPERS`` contains
the academic references supplied for the project; sources are either arXiv IDs
or stable open-access PDF URLs.
"""
import time
from pathlib import Path
from typing import Iterable

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

# Additional academic references from the project's bibliography.  Repository
# citations (DeepSORT, OC-SORT, ByteTrack, py-motmetrics, and Ultralytics) are
# intentionally not included because they are software references, not papers.
ADDITIONAL_PAPERS = {
    "1409.7618": "MOT-Literature-Review",
    "1506.0264": "YOLO",
    "2410.17725": "YOLOv11-Overview",
    "2011.11858": "GMOT-40",
    "1603.00831": "MOT16",
    "1506.01497": "Faster-R-CNN",
    "1512.02325": "SSD",
    "2005.12872": "DETR",
    "2610.19369": "Detector-Architecture-and-Scene-Density",
    "1609.01775": "Multi-Camera-Tracking-Metrics",
    "1608.08710": "Filter-Pruning",
    "1712.09867": "ShanghaiTech-Anomaly-Detection",
    "2009.07736": "HOTA",
    "1712.05877": "Quantization",
    "1807.05118": "Ray-Tune",
    "1810.05934": "ASHA",
    "2003.09003": "MOT20",
}

# Papers without a suitable arXiv identifier.
DIRECT_PAPERS = {
    "https://mdpi-res.com/d_attachment/sensors/sensors-22-02123/article_deploy/sensors-22-02123.pdf": "HOG-CNN-Multi-Camera-MOT",
    "https://openaccess.thecvf.com/content_iccv_2013/papers/Lu_Abnormal_Event_Detection_2013_ICCV_paper.pdf": "CUHK-Avenue",
}

ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "data" / "papers"


def _is_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(4) == b"%PDF"
    except OSError:
        return False


def _missing_papers(papers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return [
        (arxiv_id, name)
        for arxiv_id, name in papers
        if not _is_pdf(OUT_DIR / f"{name}.pdf")
    ]


def _missing_direct_papers(papers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return [
        (url, name)
        for url, name in papers
        if not _is_pdf(OUT_DIR / f"{name}.pdf")
    ]


def _download(url: str, name: str) -> None:
    target = OUT_DIR / f"{name}.pdf"
    print(f"downloading {name}")
    resp = requests.get(
        url,
        headers={"User-Agent": "MOT-RAG/1.0"},
        timeout=60,
    )
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise RuntimeError(f"source returned non-PDF content for {name}: {url}")
    target.write_bytes(resp.content)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arxiv_papers = {**PAPERS, **ADDITIONAL_PAPERS}
    missing_papers = _missing_papers(arxiv_papers.items())
    missing_direct_papers = _missing_direct_papers(DIRECT_PAPERS.items())
    if not missing_papers:
        print(f"all arXiv papers already exist in {OUT_DIR}")

    for arxiv_id, name in missing_papers:
        _download(f"https://arxiv.org/pdf/{arxiv_id}", name)
        time.sleep(3)  # be polite to arXiv

    for url, name in missing_direct_papers:
        _download(url, name)
        time.sleep(3)

    print(f"downloaded papers are in {OUT_DIR}")


if __name__ == "__main__":
    main()