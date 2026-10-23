# AI-Assisted Investigation Simulator

A Streamlit case investigation app for reviewing shipment evidence, requesting
restricted records, identifying conflicts, asking evidence-grounded questions,
and submitting a decision for rubric-based feedback.

## Requirements

- Windows
- Python 3.11 or later
- A Gemini API key for live interrogation and decision grading

## Setup

Open PowerShell in the project folder and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, use the virtual environment executable
directly in the commands below:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Configure Gemini

Create a file named `.env` in the project root, alongside `app.py` and
`requirements.txt`, and add your Google AI Studio key:

```dotenv
GEMINI_API_KEY=your_actual_api_key
GEMINI_MODEL=gemini-2.5-flash
```

`GEMINI_MODEL` is optional. The application defaults to `gemini-2.5-flash`.
Keep `.env` private; it is excluded from Git. Do not paste or commit your API
key into source files.

Evidence browsing, unlock validation, conflict detection, and report generation
work without a key. Interrogation and grading require a valid key and network
access to Gemini.

## Run the app

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your
browser. The public evidence is loaded at startup. Use **Request Records** to
unlock restricted documents with a justification. The **Final Decision** tab
accepts a conclusion, supporting evidence, counterevidence, confidence, and a
recommended action; it displays grading feedback and provides a Markdown
report download.

## Run tests

The normal test suite uses mocks and does not call Gemini:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_llm
```

To also run the optional live fact-extraction test, set
`RUN_LIVE_LLM_TESTS=1` in the PowerShell session and make sure `.env` contains
a valid API key:

```powershell
$env:RUN_LIVE_LLM_TESTS = "1"
.\.venv\Scripts\python.exe -m unittest -v test_llm
```

## Project files

- `app.py` — Streamlit interface and submission handling.
- `investigator.py` — session state, evidence visibility, conflict detection,
  and report assembly.
- `document_processor.py` — PDF, text, and email ingestion from `data/`.
- `llm.py` — Gemini prompts, JSON parsing, and response validation.
- `test_llm.py` — unit and Streamlit edge-case tests.
- `.streamlit/config.toml` — dark application theme.
