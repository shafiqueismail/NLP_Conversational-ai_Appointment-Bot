# extractor.py
import re, json, datetime as dt, torch
from zoneinfo import ZoneInfo
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class DentalExtractor:
    def __init__(self,
                 base_model="Qwen/Qwen2.5-0.5B-Instruct",
                 adapter_dir="./finetuned_small_dental",
                 device=None,
                 use_spacy=True):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype  = torch.float32
        self.tz     = ZoneInfo("America/Toronto")

        # optional spaCy NER for robust name extraction
        self.use_spacy = use_spacy
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")  # includes NER
            except Exception as e:
                print("spaCy not available, falling back to regex for names:", e)
                self.nlp = None

        # load tokenizer FROM adapter so vocab matches training
        self.tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True, padding_side="right")

        # base + LoRA (MPS-safe-ish)
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=self.dtype,
            attn_implementation="eager",  # avoids Apple GPU matmul issues
        )
        self.base.resize_token_embeddings(len(self.tok))
        self.base.config.pad_token_id = self.tok.pad_token_id
        self.base.config.eos_token_id = self.tok.eos_token_id
        self.base.config.use_cache = False

        self.model = PeftModel.from_pretrained(self.base, adapter_dir).eval().to(self.device)
        self.system = "You are a dental reception assistant. Always respond with JSON only—no extra text."

    # ---------- parsing helpers ----------
    @staticmethod
    def _parse_time_12h(text: str):
        m = None
        for m in re.finditer(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', text, re.I):
            pass
        if not m: return None
        hh = int(m.group(1)); mm = int(m.group(2) or 0)
        ampm = m.group(3).lower()
        if ampm == "pm" and hh != 12: hh += 12
        if ampm == "am" and hh == 12: hh = 0
        return f"{hh:02d}:{mm:02d}"

    @staticmethod
    def _parse_weekday(text: str):
        m = None
        for m in re.finditer(r'\b(next\s+)?(monday|tuesday|wednesday|thursday|friday)\b', text, re.I):
            pass
        if not m: return None, False
        return m.group(2).lower(), bool(m.group(1))

    def _resolve_date(self, weekday: str|None, said_next: bool, time_24: str|None):
        if not weekday: return None
        target = ["monday","tuesday","wednesday","thursday","friday"].index(weekday)
        now = dt.datetime.now(self.tz)
        today_wd = now.weekday()  # 0=Mon
        days_ahead = (target - today_wd) % 7
        # if "next" was said, or if same-day but the time already passed, jump a week
        if said_next or (days_ahead == 0 and time_24 and now.strftime("%H:%M") >= time_24):
            days_ahead = 7 if days_ahead == 0 else days_ahead
        date = (now + dt.timedelta(days=days_ahead)).date()
        return date.isoformat()

    @staticmethod
    def _normalize_treatment(t):
        if not t: return None
        s = t.lower()
        if "extract" in s: return "tooth extraction"
        if "fill" in s or "cavity" in s: return "cavity"
        if "clean" in s: return "cleaning"
        return s

    @staticmethod
    def _duration_for(t):
        return 90 if t and "extraction" in t else 60

    @staticmethod
    def _smart_case(name: str) -> str:
        # Title-case while preserving hyphens/apostrophes: o’connor → O’Connor, jean-luc → Jean-Luc
        def cap_seg(seg: str) -> str:
            return seg[:1].upper() + seg[1:].lower() if seg else seg
        words = []
        for w in name.strip().split():
            w = "-".join(cap_seg(p) for p in w.split("-"))
            w = "'".join(cap_seg(p) for p in w.split("'"))
            words.append(w)
        return " ".join(words)

    # NEW: robust name extraction with spaCy, with regex hints
    def _extract_name(self, text: str):
        # 1) Strong intent patterns first (capture up to 3 tokens incl. hyphens/apostrophes)
        pat = r"\b(?:my name is|name is|i am|i'm|this is)\s+([A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,2})"
        m = None
        for m in re.finditer(pat, text, re.I):
            pass
        if m:
            return self._smart_case(m.group(1))

        # 2) spaCy PERSONs (last PERSON mentioned often is the user)
        if self.nlp is not None:
            doc = self.nlp(text)
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if persons:
                return self._smart_case(persons[-1])

        # 3) Fallback mild patterns (e.g., “Mohammed Ali here”)
        m2 = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(here|speaking)\b", text)
        if m2:
            return self._smart_case(m2.group(1))

        return None

    @staticmethod
    def _infer_treatment_from_dialog(text: str):
        s = text.lower()
        if any(k in s for k in ["extract", "remove", "pull", "wisdom tooth", "tooth removal"]):
            return "tooth extraction"
        if any(k in s for k in ["fill", "filling", "cavity"]):
            return "cavity"
        if "clean" in s:
            return "cleaning"
        return None

    # ---------- main ----------
    def extract(self, dialog: str, max_new_tokens=120) -> dict:
        prompt = f"<|system|>\n{self.system}\n<|user|>\n{dialog}\n<|assistant|>\n"
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            try:
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=False,
                    eos_token_id=self.tok.eos_token_id
                )
            except Exception:
                # rare MPS hiccup → CPU fallback
                m_cpu = self.model.to("cpu").eval()
                inp_cpu = {k: v.to("cpu") for k, v in inputs.items()}
                out = m_cpu.generate(
                    **inp_cpu,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=False,
                    eos_token_id=self.tok.eos_token_id
                )

        text = self.tok.decode(out[0], skip_special_tokens=True)
        m = re.search(r"\{.*\}", text, re.S)
        obj = {}
        if m:
            try: obj = json.loads(m.group())
            except: pass

        # --- enforce user intent from dialog ---
        # Name (spaCy/regex)
        if not obj.get("name"):
            nm = self._extract_name(dialog)
            if nm: obj["name"] = nm

        # Treatment (dialog wins)
        t_dialog = self._infer_treatment_from_dialog(dialog)
        if t_dialog:
            obj["treatment"] = t_dialog
        else:
            obj["treatment"] = self._normalize_treatment(obj.get("treatment")) or "cleaning"

        # Date + time from dialog (dynamic next weekday)
        wd, said_next = self._parse_weekday(dialog)
        tm = self._parse_time_12h(dialog)
        if wd:
            obj["date"] = self._resolve_date(wd, said_next, tm or obj.get("time"))
        if tm:
            obj["time"] = tm

        # Duration by treatment
        obj["duration"] = self._duration_for(obj.get("treatment"))

        return obj
