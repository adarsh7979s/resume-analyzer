import re
from fastapi import FastAPI, UploadFile, File, Form, Request
from sentence_transformers import SentenceTransformer, util

from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import os
from dotenv import load_dotenv
from google import genai
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLE_DB_FILE = os.path.join(BASE_DIR, "role_skills_db.json")


def load_role_db():
    if not os.path.exists(ROLE_DB_FILE):
        return {}
    with open(ROLE_DB_FILE, "r") as f:
        return json.load(f)

def save_role_db(data):
    with open(ROLE_DB_FILE, "w") as f:
        json.dump(data, f, indent=2)


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
    "c sharp": ["c#", "C#"],
})


def normalize_skill(skill: str) -> str:
    skill = skill.lower().strip()

    for canonical, aliases in SKILL_ALIASES.items():
        if skill == canonical:
            return canonical
        if skill in aliases:
            return canonical

    return skill



ROLE_ALIASES = {
    "aiml": "ai engineer",
    "aiml engineer": "ai engineer",
    "ai ml": "ai engineer",
    "ai ml engineer": "ai engineer",
    "ai/ml": "ai engineer",
    "ai/ml engineer": "ai engineer",
    "ml engineer": "ai engineer",
    "machine learning engineer": "ai engineer",
}


def normalize_role(role: str) -> str:
    role = role.lower().strip()
    collapsed = role.replace("/", " ").replace("-", " ").replace("_", " ")
    collapsed = re.sub(r"\s+", " ", collapsed).strip()

    if role in ROLE_ALIASES:
        return ROLE_ALIASES[role]
    if collapsed in ROLE_ALIASES:
        return ROLE_ALIASES[collapsed]

    return collapsed


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("ðŸ”‘ Gemini key loaded:", bool(GEMINI_API_KEY))


GEMINI_ENABLED = bool(GEMINI_API_KEY)

if GEMINI_ENABLED:
    client = genai.Client(api_key=GEMINI_API_KEY)

    print("âœ… Gemini AI enabled")
else:
    print("âš ï¸ Gemini AI disabled â€” running in offline reasoning mode")





scan_history = []

# ===================================
# ðŸ§  ROLE INTELLIGENCE MEMORY
# ===================================
ROLE_INTELLIGENCE = load_role_db()


current_role = None


app = FastAPI()

# =====================================================
# ðŸ§  LOAD EMBEDDING MODEL ON STARTUP (SAFE WAY)
# =====================================================
embedding_model = None

@app.on_event("startup")
def load_embedding_model():
    global embedding_model
    print("ðŸ§  Loading semantic embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("âœ… Embedding model loaded")


# =====================================================
# ðŸ§  SEMANTIC EMBEDDING MODEL (AI UPGRADE)
# =====================================================



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
# ðŸ§  STEP 4 â€” AI CORE (ROLE â†’ CAPABILITY â†’ SKILL)
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

def evaluate_capabilities(role: str, resume_skills: list) -> dict:
    """
    Check which role capabilities are satisfied by resume skills.
    """
    capabilities = ROLE_CAPABILITIES.get(role, [])
    satisfied = []
    missing = []

    resume_set = set(resume_skills)

    for cap in capabilities:
        evidence_skills = CAPABILITY_SKILLS.get(cap, [])
        if any(skill in resume_set for skill in evidence_skills):
            satisfied.append(cap)
        else:
            missing.append(cap)

    return {
        "satisfied": satisfied,
        "missing": missing,
        "score": int((len(satisfied) / len(capabilities)) * 100) if capabilities else 0
    }





# =====================================================================
# GEMINI RESUME SKILL EXTRACTION
# =====================================================================
def gemini_extract_resume_skills(resume_text: str) -> list:
    """
    Future-proof resume skill extraction using Gemini.
    Returns structured, noise-free technical skills.
    """
    if not GEMINI_ENABLED:
        return []

    prompt = f"""
You are an expert technical resume analyzer.

TASK:
Extract ONLY concrete technical skills explicitly mentioned in the resume.

STRICT RULES:
- Ignore responsibilities, domains, concepts (e.g., "system design")
- Ignore soft skills
- Ignore adjectives (scalable, robust, distributed)
- Ignore methodologies (agile, scrum)
- NO duplicates
- NO explanations

CATEGORIES (mandatory):
- language
- framework
- library
- database
- cloud
- devops
- messaging
- tool

OUTPUT FORMAT (VALID JSON ONLY):
{{
  "skills": [
    {{ "name": "java", "category": "language" }},
    {{ "name": "spring boot", "category": "framework" }},
    {{ "name": "kafka", "category": "messaging" }}
  ]
}}

RESUME:
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

        # normalize output
        cleaned = []
        for s in skills:
            if (
                isinstance(s, dict)
                and "name" in s
                and "category" in s
                and len(s["name"].strip()) > 2
            ):
                cleaned.append({
                    "name": s["name"].lower().strip(),
                    "category": s["category"].lower().strip()
                })

        return cleaned

    except Exception as e:
        print("âš ï¸ Gemini skill extraction failed:", e)
        return []

# =====================================================================
# 1ï¸âƒ£ UPLOAD RESUME
# =====================================================================

KEYWORD_SKILL_PATTERNS = {
    "machine learning": [r"\bmachine learning\b", r"\bml\b"],
    "deep learning": [r"\bdeep learning\b"],
    "artificial intelligence": [r"\bartificial intelligence\b", r"\bai\b"],
    "python": [r"\bpython\b"],
    "sql": [r"\bsql\b"],
    "numpy": [r"\bnumpy\b"],
    "pandas": [r"\bpandas\b"],
    "scikit-learn": [r"\bscikit\s*-\s*learn\b", r"\bsklearn\b"],
    "tensorflow": [r"\btensorflow\b"],
    "pytorch": [r"\bpytorch\b"],
    "docker": [r"\bdocker\b"],
    "kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "rest api": [r"\brest\s*api\b", r"\brestful\s*api\b"],
    "graphql": [r"\bgraphql\b"],
}


def keyword_extract_resume_skills(resume_text: str) -> list:
    """
    Deterministic extractor to catch explicit skills Gemini may miss.
    """
    text = normalize_text(resume_text)
    found = []

    for skill, patterns in KEYWORD_SKILL_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            found.append(skill)

    return found


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    global resume_skills_global

    raw_text = ""

    try:
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                raw_text += page.extract_text() or ""
    except Exception:
        return {"error": "Could not read resume PDF"}
    
    normalized_text = normalize_text(raw_text)


    # Hybrid extraction: Gemini + deterministic keyword matching.
    extracted = gemini_extract_resume_skills(raw_text)
    gemini_skills = [normalize_skill(s["name"]) for s in extracted]
    keyword_skills = [normalize_skill(s) for s in keyword_extract_resume_skills(raw_text)]

    # Merge while preserving deterministic ordering for UI/debugging.
    resume_skills_global = sorted(set(gemini_skills + keyword_skills))

    return {
        "resume_skills_found": resume_skills_global,
        "total_skills_detected": len(resume_skills_global),
        "confidence": "high" if len(resume_skills_global) >= MIN_SKILLS_CONFIDENCE else "medium"
    }




    



# =====================================================================
#2ï¸âƒ£ DECIDE ON GEMINI USAGE   
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

    print("ðŸ¤– Gemini invoked for role:", role)

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

    # ðŸ” Extract JSON safely from Gemini response
        match = re.search(r"\{[\s\S]*\}", text)

        if not match:
            print("âš ï¸ Gemini response had no JSON:", text)
            return {}
        
        json_text = match.group(0)

        try:
            data = json.loads(json_text)
        except Exception as e:
            print("âš ï¸ Failed to parse Gemini JSON:", e)
            return {}
        if not isinstance(data, dict):
            return {}

        skills = data.get("skills", [])

        if not isinstance(skills, list):
            return {}

        result = {
        "role": role,
        "skills": [s.lower() for s in skills]
        }

        # ðŸ”’ SAVE role skills to local DB (cache)
        ROLE_INTELLIGENCE[role] = result["skills"]

        try:
            save_role_db(dict(ROLE_INTELLIGENCE))
        except Exception as e:
            print("âš ï¸ Failed to persist role cache:", e)



        return result


    except Exception as e:
        print("âš ï¸ Gemini failed:", e)
        return {}
    
def gemini_classify_skill_importance(role: str, skills: list) -> dict:
    """
    Classify job skills into core, supporting, and inferred (human-style).
    """
    if not GEMINI_ENABLED:
        return {}

    prompt = f"""
You are an expert technical recruiter.

JOB ROLE:
{role}

SKILLS:
{skills}

TASK:
Classify the skills into three categories:

1. core_skills â†’ must-have technical skills
2. supporting_skills â†’ good to have, learnable
3. inferred_capabilities â†’ should NOT penalize if missing

RULES:
- Do NOT invent new skills
- Use ONLY the given skills
- Return ONLY valid JSON
- No explanations

FORMAT:
{{
  "core": [],
  "supporting": [],
  "inferred": []
}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        import json, re
        match = re.search(r"\{[\s\S]*\}", response.text)
        if not match:
            return {}

        data = json.loads(match.group(0))
        return {
            "core": [s.lower() for s in data.get("core", [])],
            "supporting": [s.lower() for s in data.get("supporting", [])],
            "inferred": [s.lower() for s in data.get("inferred", [])]
        }

    except Exception as e:
        print("âš ï¸ Gemini skill classification failed:", e)
        return {}
    

# =====================================================================
# 3ï¸âƒ£ ANALYZE ROLE
# =====================================================================
# ==============================
# ðŸ§  ROLE SKILL PERSISTENT STORE
# ==============================


@app.post("/analyze-role")
async def analyze_role(request: dict):
    global job_skills_global

    

    raw_role = (
        request.get("role")
        or request.get("job_role")
        or request.get("text")
        or ""
    ).lower().strip()
    role = normalize_role(raw_role)

    global current_role
    current_role = role
    # ðŸ” Check cached role skills first
    ROLE_INTELLIGENCE.update(load_role_db())

    if role in ROLE_INTELLIGENCE:
        job_skills_global = ROLE_INTELLIGENCE[role]
        return {
            "role": role,
            "job_skills_required": job_skills_global,
            "confidence": "high",
            "decision": "cached_role"
        }



    if not role:
        return {"error": "No role received"}

    # STEP 1: Internal reasoning
    capabilities = ROLE_CAPABILITIES.get(role)
    if capabilities:
        print(f"ðŸ“¥ Role '{role}' handled internally (no Gemini)")

    # STEP 1A: Unknown role â†’ Gemini PRIMARY
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


def expand_job_skills(job_skills: list) -> list:
    """
    Convert abstract job requirements into concrete atomic skills.
    This makes matching accurate and future-proof.
    """
    expanded = set()

    for skill in job_skills:
        s = skill.lower()

        if "cloud" in s:
            expanded.update(["aws", "azure", "gcp"])

        elif "database" in s:
            expanded.update(["mysql", "postgresql", "mongodb", "sql", "nosql"])

        elif "message queue" in s or "messaging" in s:
            expanded.update(["kafka", "rabbitmq"])

        elif "programming" in s or "language" in s:
            expanded.update(["python", "java"])

        elif "container" in s:
            expanded.update(["docker", "kubernetes"])

        elif "api" in s:
            expanded.update(["rest", "graphql"])

        else:
            # fallback: keep original skill
            expanded.add(s)

    return list(expanded)


def semantic_match(resume_skills, job_skills, threshold=0.75):
    global embedding_model

    matched = []
    missing = []

    if not resume_skills or not job_skills:
        return matched, missing

    # 1️⃣ Encode everything once
    resume_embeddings = embedding_model.encode(resume_skills, convert_to_tensor=True)
    job_embeddings = embedding_model.encode(job_skills, convert_to_tensor=True)

    # 2️⃣ Compute similarity matrix
    similarity_matrix = util.cos_sim(job_embeddings, resume_embeddings)

    for i, job_skill in enumerate(job_skills):

        # 3️⃣ Exact match check (highest confidence)
        if job_skill in resume_skills:
            matched.append({
                "job_skill": job_skill,
                "matched_with": job_skill,
                "similarity": 1.0
            })
            continue

        # 4️⃣ Find best semantic match
        scores = similarity_matrix[i]
        best_score = float(scores.max())
        best_index = int(scores.argmax())
        best_resume_skill = resume_skills[best_index]

        if best_score >= threshold:
            matched.append({
                "job_skill": job_skill,
                "matched_with": best_resume_skill,
                "similarity": round(best_score, 2)
            })
        else:
            missing.append(job_skill)

    return matched, missing

       


# =====================================================================
# 3ï¸âƒ£ GET SKILL GAP
# =====================================================================
@app.get("/get-skill-gap")
async def get_skill_gap():
    global resume_skills_global, job_skills_global, scan_history, current_role

    if not resume_skills_global or not job_skills_global:
        return {"error": "Upload resume & analyze job first"}

    # 1ï¸âƒ£ Normalize resume skills
    normalized_resume = sorted(set(normalize_skill(s) for s in resume_skills_global))

    # 2ï¸âƒ£ Capability evaluation (role-based)
    capability_result = evaluate_capabilities(current_role, normalized_resume)
    capability_score = capability_result["score"]

    # 3ï¸âƒ£ Expand + normalize job skills
    expanded_job = expand_job_skills(job_skills_global)

    # keep only job skills that are relevant to resume context
    # Normalize + deduplicate job skills (do NOT filter using resume)
    normalized_job = sorted(
        set(normalize_skill(x) for x in expanded_job)
    )

    # ðŸš« filter out abstract / capability-style requirements
    ABSTRACT_TERMS = {
        "system design",
        "architecture",
        "software testing",
        "testing",
        "ci/cd",
        "version control",
        "principles",
        "web frameworks"
    }

    normalized_job = [
        s for s in normalized_job
        if not any(term in s for term in ABSTRACT_TERMS)
    ]



    # 4ï¸âƒ£ Semantic matching
    matches, missing = semantic_match(
        normalized_resume,
        normalized_job
    )

    # 5ï¸âƒ£ Match score (semantic)
    # 5️⃣ Weighted match scoring
    total_weight = 0

    for m in matches:
        sim = m["similarity"]

        if sim >= 0.90:
            total_weight += 1.0
        elif sim >= 0.80:
            total_weight += 0.85
        elif sim >= 0.75:
            total_weight += 0.70

    if normalized_job:
        base_score = (total_weight / len(normalized_job)) * 100

        # 🔻 Penalize missing skills slightly
        penalty = len(missing) * 2   # 2% penalty per missing skill

        match_score = int(max(base_score - penalty, 0))
    else:
        match_score = 0



    # 6ï¸âƒ£ Final blended score
    final_score = int(
        (match_score * 0.7) + (capability_score * 0.3)
    )

    # 7ï¸âƒ£ Extra strengths (positive skills)
    extra_strengths = sorted(
        set(normalized_resume) - set(normalized_job)
    )

    result = {
        "resume_skills": resume_skills_global,
        "job_skills_required": job_skills_global,
        "semantic_matches": matches,
        "skills_missing": missing,
        "extra_strengths": extra_strengths,
        "match_score": final_score
    }

    scan_history.append(result)
    scan_history[:] = scan_history[-10:]

    return result

    

    



# =====================================================================
@app.get("/history")
async def get_history():
    return {"history": scan_history}
