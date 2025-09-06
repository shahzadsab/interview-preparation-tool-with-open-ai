import os
import io
import time
import json
import uuid
import queue
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

from openai import OpenAI
print(os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Config ----------
MODEL_QA = "gpt-4o-mini"
TRANSCRIBE_MODEL = "whisper-1"

@dataclass
class QAItem:
    level: str          # "easy" | "medium" | "hard"
    question: str
    answer_text: str = ""
    score: float = 0.0
    feedback: str = ""

def init_state():
    if "job_desc" not in st.session_state:
        st.session_state.job_desc = ""
    if "qa" not in st.session_state:
        st.session_state.qa: List[QAItem] = []
    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame(
            columns=["timestamp", "level", "question", "score"]
        )
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = "medium"  # start here
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = None

init_state()

# ---------- Helpers ----------
def generate_questions(job_desc: str, level: str = "medium", n: int = 3) -> List[QAItem]:
    """Use GPT to generate interview questions."""
    prompt = f"""
Generate {n} interview questions for the following job description.
Tailor them to {level} difficulty. Return ONLY a JSON array of objects with a 'question' string.

Job Description:
{job_desc}
"""
    resp = client.chat.completions.create(
        model=MODEL_QA,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = resp.choices[0].message.content.strip()
    # Best-effort JSON parse
    try:
        data = json.loads(raw)
    except Exception:
        # If model returns text around JSON, extract JSON block
        start = raw.find("[")
        end = raw.rfind("]")
        data = json.loads(raw[start:end+1])

    return [QAItem(level=level, question=item["question"]) for item in data][:n]

def evaluate_answer(question: str, answer_text: str) -> Dict[str, Any]:
    """Ask GPT to evaluate the answer with a rubric and score 0–10."""
    rubric = """
You are an interview coach. Evaluate the candidate's answer on:
- Content accuracy & depth (0-10)
- Relevance to question (0-10)
- Structure & clarity (0-10)
- Verbal delivery (confidence, pace, filler words) (0-10) — infer from text.
Return JSON:
{
  "score": <0-10 overall>,
  "feedback": "<3-5 bullets with what was good & what to improve>",
  "subscores": {"content": x, "relevance": y, "structure": z, "delivery": w}
}
Keep feedback constructive and concise.
"""
    user = f"Question: {question}\nAnswer:\n{answer_text}"
    resp = client.chat.completions.create(
        model=MODEL_QA,
        messages=[
            {"role": "system", "content": rubric},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()
    # robust JSON parse
    start = raw.find("{")
    end = raw.rfind("}")
    obj = json.loads(raw[start:end+1])
    return obj

def pick_next_difficulty(history: pd.DataFrame) -> str:
    """Adaptive logic: raise/lower difficulty based on rolling average."""
    if history.empty:
        return "medium"
    recent = history.tail(5)["score"].mean()
    if recent >= 8.0:
        return "hard"
    elif recent < 5.0:
        return "easy"
    return "medium"

# ---------- Audio capture & transcription ----------
class STTAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = queue.Queue()

    def recv_audio(self, frames, sample_rate):
        # Append raw audio frames; streamlit-webrtc provides 16-bit PCM
        pcm_bytes = b"".join(f.to_ndarray().tobytes() for f in frames)
        self.audio_buffer.put((pcm_bytes, sample_rate))
        return frames

def transcribe_audio_bytes(pcm_bytes: bytes, sample_rate: int) -> str:
    """
    Send audio to OpenAI Whisper via /audio/transcriptions.
    Convert PCM to WAV in-memory for convenience.
    """
    import soundfile as sf
    import tempfile

    # Write in-memory WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # mono 16-bit PCM
        data = np.frombuffer(pcm_bytes, dtype=np.int16)
        sf.write(tmp.name, data, sample_rate, subtype="PCM_16")
        tmp.flush()
        with open(tmp.name, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL,
                file=f
            )
    return transcript.text

# ---------- UI ----------
st.title("🎤 AI Interview Preparation Coach")

with st.sidebar:
    st.header("Setup")
    st.session_state.job_desc = st.text_area(
        "Paste the target job description",
        st.session_state.job_desc,
        height=220,
        placeholder="Paste JD here…"
    )

    colA, colB = st.columns(2)
    with colA:
        base_level = st.selectbox("Start difficulty", ["easy", "medium", "hard"],
                                  index=["easy","medium","hard"].index(st.session_state.difficulty))
    with colB:
        n_q = st.number_input("Questions per round", 1, 10, 3)

    if st.button("Generate Questions", use_container_width=True, type="primary", disabled=not st.session_state.job_desc):
        st.session_state.difficulty = base_level
        st.session_state.qa = generate_questions(st.session_state.job_desc, base_level, n_q)
        st.session_state.current_idx = 0
        st.success(f"Generated {len(st.session_state.qa)} {base_level} questions.")

# Current question card
if st.session_state.qa and st.session_state.current_idx is not None:
    q = st.session_state.qa[st.session_state.current_idx]
    st.subheader(f"Q{st.session_state.current_idx+1} ({q.level.title()}): {q.question}")

    # Audio capture UI (WebRTC)
    st.write("Record your answer (click Start → speak → Stop → Transcribe).")
    ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        audio_processor_factory=STTAudioProcessor,
    )

    # Manual text fallback
    manual_answer = st.text_area("Or type your answer:", height=150)

    col1, col2, col3 = st.columns(3)
    asr_text = st.empty()

    # Buffer & transcribe
    if ctx and ctx.state.playing:
        proc: STTAudioProcessor = ctx.audio_processor
        if proc and not proc.audio_buffer.empty():
            pcm, sr = proc.audio_buffer.get()
            if st.button("Transcribe last segment"):
                with st.spinner("Transcribing…"):
                    try:
                        text = transcribe_audio_bytes(pcm, sr)
                        asr_text.write(f"**Transcribed:** {text}")
                        # append to manual box for editing
                        manual_answer = (manual_answer + " " + text).strip()
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")

    # Evaluate
    if st.button("Evaluate Answer", type="primary"):
        answer_text = manual_answer.strip()
        if not answer_text:
            st.warning("Please record or type an answer first.")
        else:
            with st.spinner("Evaluating your answer…"):
                result = evaluate_answer(q.question, answer_text)
                q.answer_text = answer_text
                q.score = float(result.get("score", 0))
                q.feedback = result.get("feedback", "")
                # Save to history
                st.session_state.history.loc[len(st.session_state.history)] = {
                    "timestamp": dt.datetime.utcnow().isoformat(),
                    "level": q.level,
                    "question": q.question,
                    "score": q.score,
                }
            st.success(f"Score: {q.score:.1f} / 10")
            st.markdown("**Feedback**")
            st.write(result["feedback"])
            # Next difficulty
            st.session_state.difficulty = pick_next_difficulty(st.session_state.history)

    # Navigation
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("Previous", disabled=st.session_state.current_idx == 0):
            st.session_state.current_idx -= 1
    with nav_col2:
        last = len(st.session_state.qa) - 1
        if st.button("Next"):
            if st.session_state.current_idx >= last:
                # Generate more at adaptive difficulty
                more = generate_questions(st.session_state.job_desc, st.session_state.difficulty, n=3)
                st.session_state.qa.extend(more)
            st.session_state.current_idx += 1

st.divider()
st.subheader("📈 Progress")

# Persist lightweight progress to disk so users can track over time
DATA_PATH = "progress.csv"
if st.button("Save progress to CSV"):
    st.session_state.history.to_csv(DATA_PATH, index=False)
    st.success(f"Saved to {DATA_PATH}")

if st.session_state.history.empty:
    st.info("No attempts yet. Once you evaluate an answer, your scores will appear here.")
else:
    df = st.session_state.history.copy()
    st.dataframe(df.tail(20), use_container_width=True)
    # Simple trend chart
    st.line_chart(df.set_index(pd.to_datetime(df["timestamp"]))["score"])


# def main():
#     print("Hello from interviewpreparationtool!")


# if __name__ == "__main__":
#     main()
