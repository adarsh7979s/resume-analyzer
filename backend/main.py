from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SKILL_SET = [
    "python","java","c++","html","css","javascript",
    "react","node","sql","mysql","mongodb",
    "machine learning","deep learning","data science",
    "django","fastapi","git","docker","unity","flutter"
]

LAST_RESUME_SKILLS = []
LAST_JOB_SKILLS = []


# ---------- Upload Resume ----------
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    global LAST_RESUME_SKILLS

    with open("temp_resume.pdf", "wb") as f:
        f.write(await file.read())

    text = ""
    with pdfplumber.open("temp_resume.pdf") as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    text = text.lower()

    LAST_RESUME_SKILLS = [s for s in SKILL_SET if s in text]

    return {
        "message": "resume processed",
        "resume_skills_found": LAST_RESUME_SKILLS
    }


# ---------- Analyze Job ----------
class JobText(BaseModel):
    text: str

@app.post("/analyze-job")
async def analyze_job(body: JobText):
    global LAST_JOB_SKILLS

    text = body.text.lower()

    LAST_JOB_SKILLS = [s for s in SKILL_SET if s in text]

    return {
        "message": "job analyzed",
        "job_skills_required": LAST_JOB_SKILLS
    }


# ---------- Get Skill Gap ----------
@app.get("/get-skill-gap")
def get_skill_gap():
    if not LAST_RESUME_SKILLS:
        return {"error": "Upload resume first"}

    if not LAST_JOB_SKILLS:
        return {"error": "Analyze job first"}

    missing = [s for s in LAST_JOB_SKILLS if s not in LAST_RESUME_SKILLS]

    score = round(
        (len(LAST_JOB_SKILLS) - len(missing)) / len(LAST_JOB_SKILLS) * 100,
        2
    )

    return {
        "resume_skills": LAST_RESUME_SKILLS,
        "job_skills_required": LAST_JOB_SKILLS,
        "skills_missing": missing,
        "match_score": score
    }
