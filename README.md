# Resume Analyzer

AI-powered resume analysis platform that:

- extracts resume skills from PDF uploads
- maps a target job role to required skills
- computes match score and ATS-style score
- highlights missing skills, strengths, and recommendations
- presents the results in a polished React dashboard

The system uses a FastAPI backend for document parsing and scoring, plus a React + Vite frontend for the user experience.

## What It Does

`Resume Analyzer` is built for a simple but useful workflow:

1. Upload a PDF resume.
2. Extract explicit skills and lightweight personal details.
3. Enter a target role such as `AI Engineer`, `Backend Engineer`, or `Frontend Developer`.
4. Infer the job's required skills using internal reasoning, cached role intelligence, and optional Gemini assistance.
5. Score the resume against the role using grouped skill matching and ATS-oriented heuristics.
6. Return recommendations, focus areas, courses, and resume upgrade suggestions.

## Core Features

- PDF resume upload and parsing with `pdfplumber`
- Deterministic skill extraction with catalog and section-based parsing
- Optional Gemini fallback for sparse resumes and unknown roles
- Semantic + lexical skill matching with `sentence-transformers`
- ATS-style score breakdown
- Analysis history tracked per session
- Interactive frontend with upload, analysis, and results views
- Configurable frontend API base URL via `VITE_API_BASE_URL`

## Tech Stack

### Backend

- `FastAPI`
- `Uvicorn`
- `pdfplumber`
- `sentence-transformers`
- `scikit-learn`
- `python-dotenv`
- `google-genai`

### Frontend

- `React 19`
- `Vite`
- `ESLint`

## Project Structure

```text
resume-analyzer/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── ai_gemini.py
│   ├── ir.py
│   ├── api/
│   ├── core/
│   ├── models/
│   └── utils/
├── frontend-react/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── components/
│   ├── package.json
│   └── .env.example
├── role_skills_db.json
└── README.md
```

## How The System Works

### 1. Resume Upload

Endpoint: `POST /upload-resume`

The backend:

- validates the uploaded file
- checks PDF size and content type
- extracts raw text from the PDF
- detects personal details like name, email, and phone
- extracts resume skills using:
  - keyword patterns
  - catalog phrase matching
  - section-aware parsing
  - optional Gemini fallback for weak deterministic coverage

### 2. Role Analysis

Endpoint: `POST /analyze-role`

The backend:

- normalizes the user-entered role
- checks built-in role capability mappings
- validates cached role intelligence from `role_skills_db.json`
- optionally asks Gemini for unknown roles
- falls back gracefully when no high-confidence inference is available

### 3. Skill Gap Scoring

Endpoint: `GET /get-skill-gap`

The backend:

- normalizes resume skills
- groups broad requirement families fairly
- compares resume skills against role requirements
- uses hybrid semantic + lexical similarity
- computes:
  - match score
  - ATS score
  - missing skills
  - extra strengths
  - recommendations

### 4. History

Endpoint: `GET /history`

Each analysis is stored in memory under an `analysis_id` and returns recent results for the current session.

## API Endpoints

### `POST /upload-resume`

Multipart form upload.

Request:

- `file`: PDF resume
- `analysis_id`: optional existing analysis session id

Response includes:

- `analysis_id`
- `resume_skills_found`
- `personal_details`
- `total_skills_detected`
- `confidence`

### `POST /analyze-role`

JSON request.

Request body example:

```json
{
  "role": "AI Engineer",
  "analysis_id": "existing-session-id"
}
```

Response includes:

- `analysis_id`
- `role`
- `job_skills_required`
- `confidence`
- `decision`

### `GET /get-skill-gap?analysis_id=<id>`

Response includes:

- `role`
- `current_role`
- `resume_skills`
- `job_skills_required`
- `semantic_matches`
- `skills_missing`
- `extra_strengths`
- `match_score`
- `ats_score`
- `ats_breakdown`
- `recommendations`

### `GET /history?analysis_id=<id>`

Returns recent analyses for the current session.

## Local Setup

### Prerequisites

- Python `3.11`
- Node.js `18+`
- npm

### Backend Setup

From the project root:

```powershell
.\.venv311_new\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Create or update `.env` in the project root:

```env
GEMINI_API_KEY=your_key_here
```

Run the backend:

```powershell
.\.venv311_new\Scripts\python.exe -m uvicorn backend.main:app --reload --reload-exclude "*.json"
```

If your `uvicorn.exe` launcher is broken after moving the project folder, reinstall it inside the current venv:

```powershell
.\.venv311_new\Scripts\python.exe -m pip install --force-reinstall uvicorn
```

### Frontend Setup

```powershell
cd frontend-react
npm install
Copy-Item .env.example .env
npm run dev
```

Default frontend env:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Environment Variables

### Backend

Important variables from `backend/main.py`:

- `GEMINI_API_KEY`
- `ROLE_CACHE_TTL_SECONDS`
- `ROLE_CACHE_MODEL_VERSION`
- `ANALYSIS_TTL_SECONDS`
- `ANALYSIS_MAX_RECORDS`
- `MAX_UPLOAD_SIZE_MB`
- `GEMINI_RESUME_MAX_CHARS`
- `GEMINI_RESUME_MIN_DETERMINISTIC`
- `ENABLE_GEMINI_ROLE_FILTER`
- `GEMINI_CONSERVE_MODE`
- `USE_GEMINI_FOR_RESUME`
- `USE_GEMINI_FOR_KNOWN_ROLE_ASSIST`
- `USE_GEMINI_FOR_UNKNOWN_ROLE`
- `GEMINI_MAX_CALLS_PER_HOUR`
- `EMBEDDING_MODEL_NAME`
- `EMBEDDING_CACHE_MAX`
- `CORS_ALLOW_ORIGINS`

### Frontend

- `VITE_API_BASE_URL`

## Development Notes

- The backend keeps live analysis state in memory.
- `role_skills_db.json` is used as a role-skill cache and can evolve over time.
- If the server restarts, in-memory analysis sessions are cleared.
- The frontend stores the current `analysis_id` in `sessionStorage`.

## Current Strengths

- The app works end-to-end locally.
- Resume extraction is more stable after cleanup of noisy composite skill phrases.
- History entries now preserve role metadata and timestamps.
- The frontend builds and passes lint cleanly.
- The API base URL is no longer hardcoded to one machine setup.

## Known Limitations

- Analysis history is in-memory, not persisted in a database.
- Resume extraction is heuristic-driven and may still miss niche or unusual skills.
- Semantic scoring quality depends on the embedding model being available.
- Gemini-backed behavior depends on a valid API key and configured usage flags.
- This is currently optimized for local/dev usage rather than production deployment.

## Troubleshooting

### `Fatal error in launcher` when starting Uvicorn

Cause:

- the Windows launcher still points to an old virtual environment path

Fix:

```powershell
.\.venv311_new\Scripts\python.exe -m pip install --force-reinstall uvicorn
.\.venv311_new\Scripts\python.exe -m uvicorn backend.main:app --reload --reload-exclude "*.json"
```

### Frontend cannot reach backend

Check:

- backend is running on port `8000`
- `frontend-react/.env` has the correct `VITE_API_BASE_URL`
- CORS is configured if using a different host

### PDF upload fails

Check:

- file is actually a PDF
- file size is within `MAX_UPLOAD_SIZE_MB`
- the PDF contains extractable text and is not only a scanned image

### Role analysis feels weak

Check:

- whether the role is covered by built-in mappings
- whether Gemini is enabled for unknown roles
- whether the role cache contains a stale or low-quality entry

## Quality Checks

Frontend:

```powershell
cd frontend-react
npm run lint
npm run build
```

Backend:

```powershell
.\.venv311_new\Scripts\python.exe -m py_compile backend\main.py
```

## Recommended Next Steps

- add automated backend tests for extraction and scoring
- split `backend/main.py` into modules for parsing, role inference, scoring, and recommendations
- persist analyses in a database instead of in-memory storage
- add Docker support for easier local and deployment workflows
- support OCR for scanned resumes

## Demo Flow

1. Start the backend.
2. Start the frontend.
3. Upload a resume PDF.
4. Enter a target role.
5. Review match score, ATS score, skill gaps, and recommendations.

---

Built to help candidates understand role fit quickly and improve resumes with more direction and less guesswork.
