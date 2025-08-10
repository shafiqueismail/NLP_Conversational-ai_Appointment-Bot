# extract.py
import argparse, json
from extractor import DentalExtractor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialog", required=True, help="Full chat transcript")
    args = ap.parse_args()

    ex = DentalExtractor()
    data = ex.extract(args.dialog)
    print(json.dumps(data, indent=2, ensure_ascii=False))
