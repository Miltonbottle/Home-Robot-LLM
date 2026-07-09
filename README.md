# 🏠🤖 Home Robot LLM Agent

A home robot you control with plain English. Type a request, and the robot figures out — step by step — how to move around, look for objects, pick them up, and bring them to you. **Safely.**

```
you> I'm thirsty, get me something to drink
Robot: I found a few things that could match: water_bottle, juice_box. Which one would you like?
you> water bottle
Robot: Here is your water bottle!
```

---

## ✨ What it does

At its core, `agent.py` runs a simple loop: at every step, it asks an LLM to choose **one** next action —

- `navigate_to` – move to a location
- `pick` – pick up an object
- `place` – put an object down
- `speak` – respond to you

The LLM only ever sees what the robot has *actually sensed so far* — it isn't handed a map or a list of objects up front. It has to explore and remember, just like a real robot would.

On top of the LLM's reasoning, two safety rules are enforced in plain code — never left to the model's judgment:

| Rule | What it means |
|---|---|
| 🧭 **Grounding** | The robot will never navigate to a made-up location or pick up an object it hasn't actually sensed. Invalid moves are rejected and the robot is told why, instead of the whole thing crashing. |
| ⚠️ **Safety** | Anything sharp, medical, chemical, or otherwise risky (e.g. `kitchen_knife`, `pill_bottle`) always triggers a confirmation question before the robot touches it. |

It also knows its own limits:
- Asks for clarification when a request is ambiguous (e.g. "get me something to drink" when there's both water *and* juice).
- Says "I can't do that" for anything outside its skills (like opening windows or flipping switches).
- Retries a few times if the gripper fails to grab an object.

**No internet? No API key? No problem.** If no LLM is configured, or a request fails, the robot automatically falls back to a small built-in planner that still follows all the same grounding and safety rules — so it never just breaks.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key

This project uses **[Groq](https://console.groq.com)** as its LLM provider — it's free to start and very fast.

1. Sign up at [console.groq.com](https://console.groq.com)
2. Create an API key
3. Install the Groq Python package:

```bash
pip install groq
```

### 3. Set your API key

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="your-key-here"
```

> 💡 No key set? The robot still works  it just uses the built-in offline planner instead of the LLM.

---

## ▶️ Running it

```bash
python run.py --test      # run the built-in test suite
python run.py              # interactive mode — type requests, type 'quit' to exit
python run.py --gui        # interactive mode with a visual pygame window
python run.py --ascii      # interactive mode with an ASCII map in the terminal
```

**Optional flags:**

| Flag | What it does |
|---|---|
| `--seed N` | Makes a run reproducible |
| `--fail 0.3` | Sets the grasp-failure probability (default: `0.3`, i.e. 30%) |

---

## 💬 Example session

```
you> I'm thirsty, get me something to drink
Robot: I sensed a few things that could match: ['water_bottle', 'juice_box']. Which one would you like?
you> water bottle
Robot: Here is your water bottle!

you> bring me the kitchen knife
Robot: 'kitchen_knife' looks like it could be unsafe to handle (sharp_utensil). Are you sure you'd like me to bring it?
you> yes
Robot: Done — I've taken care of that.
```


