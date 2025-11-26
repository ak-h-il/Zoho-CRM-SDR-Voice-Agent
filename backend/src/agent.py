# backend/src/agent.py
import logging
import os
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
load_dotenv(".env.local")

# -------------------------
# Paths / data dirs (FIXED)
# -------------------------
SRC_DIR = Path(__file__).resolve().parent            # backend/src/
DATA_DIR = SRC_DIR / "data"                         # backend/src/data/
FAQ_PATH = DATA_DIR / "zoho_crm_faq.json"
LEADS_DIR = DATA_DIR / "leads"
MEETINGS_DIR = DATA_DIR / "meetings"
CALENDAR_PATH = DATA_DIR / "mock_calendar.json"

# Ensure folders exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
LEADS_DIR.mkdir(exist_ok=True, parents=True)
MEETINGS_DIR.mkdir(exist_ok=True, parents=True)

# -------------------------
# Global session
# -------------------------
SESSION = None

VOICE_MAP = {
    "default": "en-US-alicia",
    "answer": "en-US-alicia",
    "confirm": "en-US-alicia",
    "booking": "en-US-alicia"
}

# -------------------------
# Load helper files
# -------------------------
def load_faq():
    if not FAQ_PATH.exists():
        raise FileNotFoundError(
            f"zoho_crm_faq.json missing at: {FAQ_PATH}"
        )
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def tokenize_text(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return [t for t in s.split() if len(t) > 1]

def find_faq_answer(user_text: str, faq_data: dict):
    tokens = set(tokenize_text(user_text))
    best = None
    best_score = 0
    for entry in faq_data.get("faqs", []):
        combo = entry["q"] + " " + entry["a"]
        ctoks = set(tokenize_text(combo))
        score = len(tokens.intersection(ctoks))
        if score > best_score:
            best_score = score
            best = entry
    return best if best_score > 0 else None

def save_json_atomic(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)

def save_lead(lead: dict):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    p = LEADS_DIR / f"lead_{ts}.json"
    save_json_atomic(p, lead)
    return str(p)

def save_meeting(meeting: dict):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    p = MEETINGS_DIR / f"meeting_{ts}.json"
    save_json_atomic(p, meeting)
    return str(p)

def load_or_init_calendar():
    if CALENDAR_PATH.exists():
        return json.load(open(CALENDAR_PATH, "r", encoding="utf-8"))
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    slots = []
    for i in range(1, 6):
        t = now + timedelta(hours=24 + i)
        slots.append({
            "id": f"slot-{i}",
            "start": t.isoformat() + "Z",
            "duration_minutes": 30,
            "booked": False
        })
    save_json_atomic(CALENDAR_PATH, slots)
    return slots

# -------------------------
# SDR Agent
# -------------------------
class SdrAgent(Agent):
    def __init__(self):
        self.faq = load_faq()

        self.lead = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None,
            "notes": None
        }

        self.meeting = None
        self.calendar = load_or_init_calendar()

        INSTRUCTIONS = """
You are "Asha", a warm, friendly, and efficient Sales Development Representative (SDR) for Zoho CRM.

Your behavior:
- Greet warmly, ask what the caller is exploring.
- Understand their business, team size, workflow, and challenges.
- ALWAYS answer product, feature, or pricing questions ONLY using the FAQ tool (answer_faq).
- NEVER invent features or numbers.
- Collect lead information naturally using collect_lead(field, value):
  name, company, email, role, use_case, team_size, timeline.
  Confirm each value softly: “Got it — your email is ___, right?”
- Offer a meeting when user shows interest:
  Use list_slots() → show 2–3 options.
  When user chooses, call book_slot(slot_id).
- Detect call endings: “that’s all”, “thanks”, “bye”, “done”.
  Then call finalize_and_save(reason) and read the summary to the user.

Tone:
- Very friendly.
- Never robotic.
- One question at a time.
"""

        super().__init__(instructions=INSTRUCTIONS)

    # -------------------------
    # Tools
    # -------------------------
    @function_tool
    async def answer_faq(self, ctx: RunContext, query: str):
        e = find_faq_answer(query, self.faq)
        if not e:
            return {"ok": False, "error": "No matching FAQ found."}
        return {"ok": True, "question": e["q"], "answer": e["a"]}

    @function_tool
    async def collect_lead(self, ctx: RunContext, field: str, value: str):
        if field not in self.lead:
            return {"ok": False, "error": f"Unknown field {field}"}
        self.lead[field] = value.strip()
        return {"ok": True, "field": field, "value": self.lead[field]}

    @function_tool
    async def get_lead(self, ctx: RunContext):
        return {"ok": True, "lead": self.lead, "meeting": self.meeting}

    @function_tool
    async def list_slots(self, ctx: RunContext):
        slots = [s for s in self.calendar if not s["booked"]]
        return {"ok": True, "slots": slots}

    @function_tool
    async def book_slot(self, ctx: RunContext, slot_id: str):
        for s in self.calendar:
            if s["id"] == slot_id:
                if s["booked"]:
                    return {"ok": False, "error": "Slot already booked."}
                s["booked"] = True
                save_json_atomic(CALENDAR_PATH, self.calendar)

                self.meeting = {
                    "slot": slot_id,
                    "start": s["start"],
                    "duration_minutes": s["duration_minutes"],
                    "lead": self.lead.copy(),
                    "booked_at": datetime.utcnow().isoformat() + "Z"
                }
                save_meeting(self.meeting)
                return {"ok": True, "meeting": self.meeting}

        return {"ok": False, "error": "Slot ID not found"}

    @function_tool
    async def finalize_and_save(self, ctx: RunContext, reason: str = ""):
        lead_copy = self.lead.copy()
        lead_copy["saved_at"] = datetime.utcnow().isoformat() + "Z"
        path = save_lead(lead_copy)

        meeting_path = None
        if self.meeting:
            meeting_path = save_meeting(self.meeting)

        summary = (
            f"Lead saved: {lead_copy.get('name','Unknown')} from {lead_copy.get('company','Unknown')} "
            f"({lead_copy.get('role','Unknown')}). Use case: {lead_copy.get('use_case','')}. "
            f"Timeline: {lead_copy.get('timeline','')}. "
        )
        if self.meeting:
            summary += f"Meeting booked at {self.meeting['start']}."

        return {
            "ok": True,
            "summary": summary,
            "lead_file": path,
            "meeting_file": meeting_path
        }

# -------------------------
# prewarm VAD
# -------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# -------------------------
# entrypoint
# -------------------------
async def entrypoint(ctx: JobContext):
    global SESSION
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice=VOICE_MAP["default"],
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    SESSION = session
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_mc(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def on_shutdown():
        logger.info(f"Usage summary: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(on_shutdown)

    agent = SdrAgent()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )

    await ctx.connect()

# -------------------------
# run
# -------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
