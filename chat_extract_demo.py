# chat_extract_demo.py
import json
from extractor import DentalExtractor
from extract import next_prompt, REQUIRED_FIELDS  # reuse your prompt logic

def format_dialog(history):
    # Turns [{"role":"user","content":"..."}, ...] into the transcript string
    lines = []
    for m in history:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)

def main():
    print("Interactive dental intake • type 'exit' to quit\n")
    ex = DentalExtractor()  # uses MPS if available

    history = []
    # initial user turn
    first = input("You: ").strip()
    if not first or first.lower() in ("exit", "quit"):
        return
    history.append({"role": "user", "content": first})

    while True:
        dialog = format_dialog(history)
        data = ex.extract(dialog)

        missing = [k for k in REQUIRED_FIELDS if not data.get(k)]
        if missing:
            prompt = next_prompt(missing, data)
            print(f"Assistant: {prompt}")
            history.append({"role": "assistant", "content": prompt})

            msg = input("You: ").strip()
            if msg.lower() in ("exit", "quit"):
                print("Assistant: Got it — ending the session.")
                break
            history.append({"role": "user", "content": msg})
            continue

        # all fields present -> show summary and allow confirm or edits
        print("Assistant: You're all set! Here are the details:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("Assistant: Type 'confirm' to accept, or add any change (e.g., 'make it 2:30pm').")

        reply = input("You: ").strip()
        if reply.lower() in ("confirm", "yes", "y", "ok"):
            print("Assistant: Booking confirmed (simulated). 🎉")
            break
        if reply.lower() in ("exit", "quit"):
            print("Assistant: Ending without confirming.")
            break
        # treat reply as another user turn and loop again
        history.append({"role": "user", "content": reply})

if __name__ == "__main__":
    main()

