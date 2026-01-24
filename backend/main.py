import re
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import os
from dotenv import load_dotenv
from google import genai

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9\s#+]", " ", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


MIN_SKILLS_CONFIDENCE = 5


BLOCKED_SKILLS = {
    "skills",
    "skill",
    "confidence",
    "resume",
    "resume skills",
    "resume_skills_found",
    "total_skills_detected",
    "technology",
    "technologies",
    "tools",
    "tool",
}

SKILL_ALIASES = {
    "unity engine": ["unity", "unity3d"],
    "unreal engine": ["unreal", "ue4", "ue5"],
    "c plus plus": ["c++"],
    "data structures and algorithms": ["dsa", "data structures", "algorithms"],
    "machine learning": ["ml"],
}
SKILL_ALIASES.update({
    "artificial intelligence": ["ai"],
    "virtual reality": ["vr"],
    "rest api": ["rest", "restful api"],
    "android studio": ["android"],
    "visualization": ["data visualization"]
})
SKILL_ALIASES.update({
    "figma": ["figma design", "ui figma"],
    "virtual reality": ["vr"],
    "augmented reality": ["ar"],
    "android studio": ["android"],
    "data visualization": ["visualization", "viz"],
})
SKILL_ALIASES.update({
    "c sharp": ["c#"]
})


def normalize_skill(skill: str) -> str:
    skill = skill.lower().strip()

    for canonical, aliases in SKILL_ALIASES.items():
        if skill == canonical:
            return canonical
        if skill in aliases:
            return canonical

    return skill



load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("🔑 Gemini key loaded:", bool(GEMINI_API_KEY))


GEMINI_ENABLED = bool(GEMINI_API_KEY)

if GEMINI_ENABLED:
    client = genai.Client(api_key=GEMINI_API_KEY)

    print("✅ Gemini AI enabled")
else:
    print("⚠️ Gemini AI disabled — running in offline reasoning mode")





scan_history = []

app = FastAPI()

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- TEMP STORAGE ----------
resume_skills_global = []
job_skills_global = []

# =====================================================
# 🧠 STEP 4 — AI CORE (ROLE → CAPABILITY → SKILL)
# =====================================================

ROLE_CAPABILITIES = {
    "game developer": [
        "gameplay programming",
        "real-time rendering",
        "game physics",
        "asset integration"
    ],

    "electrical engineer": [
        "circuit design",
        "power systems",
        "control systems",
        "industrial automation"
    ],

    "ai engineer": [
        "model development",
        "data processing",
        "model deployment",
        "algorithm optimization"
    ],

    "full stack developer": [
        "frontend development",
        "backend development",
        "api integration",
        "database design"
    ],

    "frontend developer": [
    "frontend development",
    "ui development",
    "web performance",
    "api consumption"
   ],
   "ui development": [
    "responsive design",
    "accessibility",
    "css frameworks"
],

"web performance": [
    "lazy loading",
    "bundling",
    "optimization"
],

"api consumption": [
    "rest",
    "json",
    "fetch",
    "axios"
]

}

CAPABILITY_SKILLS = {
    "gameplay programming": ["c#", "unity"],
    "real-time rendering": ["unity", "shader programming"],
    "game physics": ["physics engines", "c#"],
    "asset integration": ["blender", "3d modeling"],

    "circuit design": ["matlab", "pcb design"],
    "power systems": ["power electronics", "simulink"],
    "control systems": ["plc", "automation"],
    "industrial automation": ["scada", "plc"],

    "model development": ["python", "machine learning", "deep learning"],
    "data processing": ["pandas", "numpy", "sql"],
    "model deployment": ["docker", "apis"],
    "algorithm optimization": ["python"],

    "frontend development": ["html", "css", "javascript", "react"],
    "backend development": ["python", "fastapi", "apis"],
    "api integration": ["rest", "json"],
    "database design": ["sql", "mysql"]
}



# =====================================================================
# 1️⃣ UPLOAD RESUME
# =====================================================================

# ---------- GLOBAL SKILL UNIVERSE ----------
GLOBAL_SKILLS = set()
GLOBAL_SKILLS.update({
    "figma",
    "virtual reality",
    "augmented reality",
    "ui ux",
    "user interface",
    "user experience"
})
GLOBAL_SKILLS.add("c#")



# Add capability skills
for skills in CAPABILITY_SKILLS.values():
    for skill in skills:
        GLOBAL_SKILLS.add(skill.lower())

# Add aliases
for canonical, aliases in SKILL_ALIASES.items():
    GLOBAL_SKILLS.add(canonical)
    for a in aliases:
        GLOBAL_SKILLS.add(a)



# =====================================================================
# GEMINI RESUME SKILL EXTRACTION
# =====================================================================
def gemini_extract_resume_skills(resume_text: str) -> list:
    """
    Uses Gemini ONLY to extract skills from resume text.
    Returns a list of skills.
    """
    if not GEMINI_ENABLED:
        return []

    prompt = f"""
You are an expert resume parser.

Task:
Extract ONLY technical skills from the resume text below.

Rules:
- Return ONLY valid JSON
- No explanations
- No markdown
- No duplicates
- Skills should be short (1–3 words)

JSON format:
{{
  "skills": ["skill1", "skill2", "skill3"]
}}

Resume Text:
\"\"\"
{resume_text[:12000]}
\"\"\"
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        import json, re

        match = re.search(r"\{[\s\S]*\}", response.text)
        if not match:
            return []

        data = json.loads(match.group(0))
        skills = data.get("skills", [])

        if not isinstance(skills, list):
            return []

        return [
            s.lower().strip()
            for s in skills
            if isinstance(s, str) and len(s) > 2
        ]





    except Exception as e:
        print("⚠️ Gemini resume skill extraction failed:", e)
        return []
# =====================================================================
# 1️⃣ UPLOAD RESUME
# =====================================================================


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    global resume_skills_global

    raw_text = ""

    # 1️⃣ Read PDF
    try:
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                raw_text += page.extract_text() or ""
    except Exception:
        return {"error": "Could not read resume PDF"}

    normalized_text = normalize_text(raw_text)

    # 2️⃣ RULE-BASED SKILL DETECTION
    detected = []

    for skill in GLOBAL_SKILLS:
        if len(skill) < 3:
            continue  # avoids garbage like ai, ml, it

        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, normalized_text):
            detected.append(skill)

    # normalize aliases
    detected = list(set(normalize_skill(s) for s in detected))

    # 3️⃣ GEMINI AUGMENTATION (adds missing skills like figma, vr, tools)
    if GEMINI_ENABLED:
        gemini_skills = gemini_extract_resume_skills(raw_text)

        detected = sorted(
            set(detected) | set(normalize_skill(s) for s in gemini_skills)
        )

    # 4️⃣ FINAL CLEAN FILTER (anti-garbage)
    detected = [
    s for s in detected
    if len(s) > 2 and s not in BLOCKED_SKILLS
]


    resume_skills_global = detected

    # 🔒 FINAL /upload-resume RETURN (LOCKED)
    return {
        "resume_skills_found": resume_skills_global,
        "total_skills_detected": len(resume_skills_global),
        "confidence": (
            "high" if len(resume_skills_global) >= MIN_SKILLS_CONFIDENCE
            else "medium"
        )
    }



    



# =====================================================================
#2️⃣ DECIDE ON GEMINI USAGE   
#====================================================================


#gemini_infer_role_skills

def should_use_gemini(role: str, skills: list) -> bool:
    """
    Decide whether Gemini should be used.
    """
    if not skills:
        return True

    if len(skills) < MIN_SKILLS_CONFIDENCE:
        return True

    if role not in ROLE_CAPABILITIES:
        return True

    return False

def gemini_infer_role_skills(role: str) -> dict:
    """
    Ask Gemini ONLY to infer skills for a job role.
    Returns structured JSON or empty result on failure.
    """
    if not GEMINI_ENABLED:
        return {}

    print("🤖 Gemini invoked for role:", role)

    prompt = f"""
You are an expert AI system.

Task:
Given a job role, list the core technical skills required.

Rules:
- Return ONLY valid JSON
- No explanations
- No markdown
- No extra text

JSON format:
{{
  "role": "{role}",
  "skills": ["skill1", "skill2", "skill3"]
}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        import json
        import re

        text = response.text.strip()

    # 🔐 Extract JSON safely from Gemini response
        match = re.search(r"\{[\s\S]*\}", text)

        if not match:
            print("⚠️ Gemini response had no JSON:", text)
            return {}
        
        json_text = match.group(0)

        try:
            data = json.loads(json_text)
        except Exception as e:
            print("⚠️ Failed to parse Gemini JSON:", e)
            return {}
        if not isinstance(data, dict):
            return {}

        skills = data.get("skills", [])

        if not isinstance(skills, list):
            return {}

        return {
            "role": role,
            "skills": [s.lower() for s in skills]
        }

    except Exception as e:
        print("⚠️ Gemini failed:", e)
        return {}

# =====================================================================
# 3️⃣ ANALYZE ROLE
# =====================================================================
@app.post("/analyze-role")
async def analyze_role(request: dict):
    global job_skills_global

    print("📥 RAW REQUEST:", request)

    role = (
        request.get("role")
        or request.get("job_role")
        or request.get("text")
        or ""
    ).lower().strip()

    if not role:
        return {"error": "No role received"}

    # STEP 1: Internal reasoning
    capabilities = ROLE_CAPABILITIES.get(role)

    # STEP 1A: Unknown role → Gemini PRIMARY
    if not capabilities:
        if GEMINI_ENABLED:
            gemini_result = gemini_infer_role_skills(role)

            if (
                gemini_result.get("skills")
                and len(gemini_result["skills"]) >= MIN_SKILLS_CONFIDENCE
            ):
                job_skills_global = sorted(set(gemini_result["skills"]))
                return {
                    "role": role,
                    "job_skills_required": job_skills_global,
                    "confidence": "high",
                    "decision": "gemini_primary"
                }

        return {
            "error": f"Role '{role}' not recognized",
            "confidence": "none",
            "decision": "rejected"
        }

    # STEP 2: Internal skill inference
    skills = []
    for cap in capabilities:
        skills.extend(CAPABILITY_SKILLS.get(cap, []))

    skills = sorted(set(skills))

    # STEP 3: Confidence decision
    use_gemini = should_use_gemini(role, skills)

    if use_gemini and GEMINI_ENABLED:
        gemini_result = gemini_infer_role_skills(role)

        if (
            gemini_result.get("skills")
            and len(gemini_result["skills"]) >= MIN_SKILLS_CONFIDENCE
        ):
            job_skills_global = sorted(set(gemini_result["skills"]))
            return {
                "role": role,
                "job_skills_required": job_skills_global,
                "confidence": "high",
                "decision": "gemini_assisted"
            }

    # STEP 4: Final fallback (internal)
    job_skills_global = skills
    return {
        "role": role,
        "job_skills_required": skills,
        "confidence": "high" if len(skills) >= MIN_SKILLS_CONFIDENCE else "low",
        "decision": "internal_reasoning"
    }






# =====================================================================
# 3️⃣ GET SKILL GAP
# =====================================================================
@app.get("/get-skill-gap")
async def get_skill_gap():
    global resume_skills_global, job_skills_global, scan_history

    if not resume_skills_global or not job_skills_global:
        return {"error": "Upload resume & analyze job first"}

    normalized_resume = {normalize_skill(s) for s in resume_skills_global}
    normalized_job = {normalize_skill(s) for s in job_skills_global}

    matched = sorted(normalized_resume & normalized_job)
    missing = sorted(normalized_job - normalized_resume)
    score = int(
        (len(job_skills_global) - len(missing)) /
        len(job_skills_global) * 100
    ) if job_skills_global else 0

    result = {
    "resume_skills": sorted(normalized_resume),
    "job_skills_required": sorted(normalized_job),
    "skills_matched": matched,
    "skills_missing": missing,
    "match_score": int(len(matched) / len(normalized_job) * 100)
}


    # SAVE TO HISTORY (keep last 10 only)
    scan_history.append(result)
    scan_history[:] = scan_history[-10:]

    return result
# =====================================================================
@app.get("/history")
async def get_history():
    return {"history": scan_history}