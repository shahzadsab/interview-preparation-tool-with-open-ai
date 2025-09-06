#  AI Interview Preparation Coach

An interactive **Streamlit** app to help users practice job interviews using AI. Users input a job description, receive dynamically generated interview questions, respond verbally, and get feedback — all while tracking progress over time.

##  Demo

*Workflow: Job description → AI-generated questions → spoken response → AI feedback → skill tracking.*

---

##  Features

- **Custom Question Generation**  
  Leverages OpenAI GPT to create interview questions based on your target job.

- **Speech-to-Text Practice**  
  Use your microphone to verbally answer; the app transcribes with Whisper and supports typed responses.

- **AI Evaluation & Feedback**  
  GPT scores your answer (0–10) across content, relevance, structure, and delivery, and gives actionable feedback.

- **Adaptive Difficulty**  
  Difficulty adjusts (easy → medium → hard) based on recent performance trends.

- **Progress Tracking**  
  Stores your history (score, question, timestamp) in-session and allows export to CSV. Includes trend charts.

---

##  Getting Started

### Requirements

- Python 3.10+
- OpenAI API key (set the `OPENAI_API_KEY` environment variable)

### Install Dependencies

```bash
source .venv/bin/activate
uv sync
uv run streamlit run main.py
```



