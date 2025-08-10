import json, re, argparse
from datetime import datetime

WEEKDAY_TO_DATE = {  # matches the dates used in your data
    "monday":  "2025-08-04",
    "tuesday": "2025-08-05",
    "wednesday":"2025-08-06",
    "thursday":"2025-08-07",
    "friday":  "2025-08-01",
}
VALID_TREATMENTS = {"cleaning","filling","extraction"}

def canonical_treatment(s: str):
    s = (s or "").lower().strip()
    if "clean" in s:
        return "cleaning"
    if "cavity" in s or "fill" in s:
        return "filling"
    if "extract" in s or "remove" in s or "tooth extraction" in s:
        return "extraction"
    return s

def to_minutes(duration):
    if duration is None:
        return None
    if isinstance(duration, (int, float)):
        return int(duration)
    s = str(duration).lower()
    if "1.5" in s or "90" in s:
        return 90
    if "1" in s and "hour" in s:
        return 60
    if "60" in s:
        return 60
    return None

def parse_name_from_prompt(prompt: str):
    # Most common patterns: "User: Sarah Ali.", "User: My name is Mohammed Ali"
    # Grab the last name-like phrase
    candidates = re.findall(r"User:\s*(?:my name is\s*)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\.?", prompt, re.I)
    return candidates[-1].strip() if candidates else None

def parse_weekday_from_prompt(prompt: str):
    m = re.search(r"\b(monday|tuesday|wednesday|thursday|friday)\b", prompt, re.I)
    return m.group(1).lower() if m else None

def parse_time_from_prompt(prompt: str):
    # Handles "2pm", "11:30am", "3", "3 pm", "14:00", "2PM", "half past two" (ignored)
    p = prompt.lower()
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", p)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or 0)
        ampm = m.group(3)
        if ampm == "pm" and hh != 12: hh += 12
        if ampm == "am" and hh == 12: hh = 0
        return f"{hh:02d}:{mm:02d}"
    # 24h like 14:00
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", p)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"
    # Bare hour like "at 3", "around 2"
    m = re.search(r"\b(?:at|around)?\s*\b([1-9]|1[0-2])\b(?!\s*[:\d])", p)
    if m:
        # Ambiguous → assume afternoon booking (clinic hours); set to 14:00 if 2, etc.
        hh = int(m.group(1))
        if 9 <= hh <= 17:
            return f"{hh:02d}:00"
        # bias to afternoon (common in your data)
        return f"{(hh%12)+12:02d}:00"
    return None

def normalize_time_str(t):
    if not t: return None
    t = t.strip()
    # already HH:MM?
    if re.fullmatch(r"[0-2]\d:[0-5]\d", t):
        return t
    # 9am/2pm etc.
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)", t.lower())
    if m:
        hh = int(m.group(1)); mm = int(m.group(2) or 0); ampm = m.group(3)
        if ampm == "pm" and hh != 12: hh += 12
        if ampm == "am" and hh == 12: hh = 0
        return f"{hh:02d}:{mm:02d}"
    return t  # fallback

def normalize_record(prompt, completion_obj, align_weekday: bool):
    # 1) Treatment
    if "treatment" in completion_obj:
        completion_obj["treatment"] = canonical_treatment(completion_obj["treatment"])
        if completion_obj["treatment"] not in VALID_TREATMENTS:
            # leave as-is but you may want to drop later
            pass

    # 2) Name: prefer name in prompt if present
    name_in_prompt = parse_name_from_prompt(prompt)
    if name_in_prompt:
        completion_obj["name"] = name_in_prompt

    # 3) Time: prefer prompt time if present; else normalize existing
    time_in_prompt = parse_time_from_prompt(prompt)
    if time_in_prompt:
        completion_obj["time"] = time_in_prompt
    elif "time" in completion_obj:
        completion_obj["time"] = normalize_time_str(completion_obj["time"])

    # 4) Date: optionally align to weekday from prompt
    if align_weekday:
        wd = parse_weekday_from_prompt(prompt)
        if wd and wd in WEEKDAY_TO_DATE:
            completion_obj["date"] = WEEKDAY_TO_DATE[wd]

    # 5) Duration: number of minutes
    minutes = to_minutes(completion_obj.get("duration"))
    if minutes is None:
        # default by treatment
        t = completion_obj.get("treatment")
        if t == "extraction":
            minutes = 90
        else:
            minutes = 60
    completion_obj["duration"] = int(minutes)

    # 6) Payments unified when present
    if "payment_methods" in completion_obj:
        completion_obj["payment_methods"] = ["cash", "debit"]

    # 7) Hours/price: keep as-is if present (already consistent in your set)
    # 8) Sort keys for readability (optional)
    order = ["name","treatment","date","time","duration","price","hours","closed_on","payment_methods"]
    completion_obj = {k: completion_obj[k] for k in order if k in completion_obj}

    return completion_obj

def load_any(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if txt.startswith("["):
            return json.loads(txt)
        # try JSONL
        data = []
        for line in txt.splitlines():
            line = line.strip()
            if not line: continue
            data.append(json.loads(line))
        return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input dataset (array of {prompt,completion})")
    ap.add_argument("--out", dest="out", required=True, help="Output cleaned dataset (same schema)")
    ap.add_argument("--align-weekday", action="store_true", help="Align date to weekday mentioned in prompt")
    args = ap.parse_args()

    raw = load_any(args.inp)

    # dedupe identical (prompt, completion) pairs
    seen = set()
    rows = []
    for r in raw:
        key = (r.get("prompt","").strip(), r.get("completion","").strip())
        if key in seen: 
            continue
        seen.add(key)
        rows.append(r)

    cleaned = []
    bad = 0
    for r in rows:
        prompt = r.get("prompt","")
        comp_txt = r.get("completion","").strip()

        # Extract JSON block from completion
        m = re.search(r"\{.*\}", comp_txt, re.S)
        if not m:
            # skip items with no JSON completion
            bad += 1
            continue
        try:
            comp = json.loads(m.group())
        except Exception:
            bad += 1
            continue

        comp_clean = normalize_record(prompt, comp, align_weekday=args.align_weekday)
        cleaned.append({"prompt": prompt.strip(), "completion": json.dumps(comp_clean, ensure_ascii=False)})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Cleaned: {len(cleaned)} items. Skipped (couldn't parse): {bad}.")

if __name__ == "__main__":
    main()

