# Local Film Lab Setup (Windows)

The Film Lab must run locally while the source game video lives on your PC.
A deployed Streamlit Cloud app cannot directly read a Windows path such as
`C:\Football\NorthTexas_Tulsa.mp4`.

## 1. Clone the repository

```powershell
git clone https://github.com/watkinspalmer-sys/cfb-scouting.git
cd cfb-scouting
git checkout video-analysis-v1
```

If you already cloned the repository:

```powershell
git fetch
git checkout video-analysis-v1
git pull
```

## 2. Create a Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Add your CFBD API key locally

Create:

`.streamlit\secrets.toml`

with:

```toml
CFBD_API_KEY = "your-key-here"
```

Never commit this file. It is already ignored by Git.

## 4. Verify FFmpeg

```powershell
ffmpeg -version
```

## 5. Run Streamlit locally

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints, usually:

`http://localhost:8501`

The sidebar will include the Film Lab page.

## Local data

- Full game broadcasts stay on your PC.
- Extracted clips go into `clips/`.
- Reviewed charting data goes into `local_data/film_chart.csv`.
- These local artifacts are not committed to GitHub.


## 6. Configure AI Analyzer

AI Analyzer v1 uses the OpenAI Responses API with sampled JPEG frames from each
short snap clip. The full game broadcast stays on your PC.

Add this to the same local file:

`.streamlit\secrets.toml`

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
OPENAI_MODEL = "gpt-6-luna"
```

Do not commit or share the API key.

The OpenAI API is billed separately from a ChatGPT subscription. Configure API
billing/credits in your OpenAI API account before running AI analysis.

After changing requirements, install them with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then start the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **AI Analyzer** in the Streamlit sidebar. Select one of the reviewed
benchmark snaps, confirm whether the chart team is on offense or defense, and
click **Analyze snap with AI**.

AI predictions are stored locally in:

`local_data\ai_predictions.csv`

Temporary sampled frames are stored in `tmp_ai_frames\` and are ignored by
Git.
