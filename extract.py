# extract.py
import argparse, json, sys
from extractor import DentalExtractor

REQUIRED_FIELDS = ["name", "treatment", "date", "time"]

def summarize_known(data: dict) -> str:
    parts = []
    if data.get("treatment"):
        parts.append(data["treatment"])
    if data.get("date") and data.get("time"):
        parts.append(f"on {data['date']} at {data['time']}")
    elif data.get("date"):
        parts.append(f"on {data['date']}")
    elif data.get("time"):
        parts.append(f"at {data['time']}")
    if data.get("name"):
        parts.append(f"for {data['name']}")
    return ", ".join(parts)

def next_prompt(missing: list[str], data: dict) -> str | None:
    if not missing:
        return None
    templates = {
        "name": "What’s your full name?",
        "treatment": "What treatment do you need—cleaning, cavity filling, or tooth extraction?",
        "date": "Which weekday works for you (Mon–Fri) or what date would you like?",
        "time": "What time works for you (e.g., 10:00am, 2:30pm)?",
    }
    known = summarize_known(data)
    prefix = f"Great — I have {known}. " if known else ""
    if len(missing) == 1:
        return prefix + templates[missing[0]]
    if len(missing) == 2:
        a, b = missing
        return prefix + f"I just need two things to finish the booking: {templates[a]} {templates[b]}"
    labels = {
        "name": "your full name",
        "treatment": "the treatment",
        "date": "the date",
        "time": "the time",
    }
    list_text = ", ".join(labels[m] for m in missing[:-1]) + f", and {labels[missing[-1]]}"
    return prefix + f"To finish the booking, could you share {list_text}?"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialog", required=True, help="Full chat transcript")
    ap.add_argument("--strict", action="store_true",
                    help="Exit with code 1 if any required field is missing")
    args = ap.parse_args()

    ex = DentalExtractor()
    data = ex.extract(args.dialog)

    missing = [k for k in REQUIRED_FIELDS if not data.get(k)]
    ok = len(missing) == 0
    prompt = next_prompt(missing, data)

    print(json.dumps({"ok": ok, "data": data, "missing": missing, "next_prompt": prompt},
                     indent=2, ensure_ascii=False))

    if args.strict and not ok:
        sys.exit(1)
