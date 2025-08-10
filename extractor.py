# extractor.py
import re, json, datetime as dt, torch
from zoneinfo import ZoneInfo
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class DentalExtractor:
    def __init__(self,
                 base_model="Qwen/Qwen2.5-0.5B-Instruct",
                 adapter_dir="./finetuned_small_dental",
                 device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype  = torch.float32
        self.tz     = ZoneInfo("America/Toronto")

        # load tokenizer FROM adapter so vocab matches training
        self.tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True, padding_side="right")

        # base + LoRA (MPS-safe)
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

    # --- parsing helpers ---
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
        # last weekday mentioned; captures "next Monday" too
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

    # --- main ---
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

        # post-fix with user intent
        wd, said_next = self._parse_weekday(dialog)
        tm = self._parse_time_12h(dialog)
        if wd:
            obj["date"] = self._resolve_date(wd, said_next, tm or obj.get("time"))
        if tm:
            obj["time"] = tm
        obj["treatment"] = self._normalize_treatment(obj.get("treatment")) or "cleaning"
        obj["duration"] = self._duration_for(obj["treatment"])
        return obj
