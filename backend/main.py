import re
from fastapi import FastAPI, UploadFile, File, Form, Request
from sentence_transformers import SentenceTransformer, util

from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import os
from dotenv import load_dotenv
from google import genai
import json
import shutil
import tempfile
import threading
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4
from difflib import SequenceMatcher
import importlib
import torch

_RAPIDFUZZ_FUZZ = None
_RAPIDFUZZ_CHECKED = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLE_DB_FILE = os.path.join(BASE_DIR, "role_skills_db.json")
ROLE_DB_LOCK = threading.Lock()
ANALYSIS_STORE_LOCK = threading.Lock()
ANALYSIS_STORE: Dict[str, Dict[str, Any]] = {}
ROLE_CACHE_TTL_SECONDS = int(os.getenv("ROLE_CACHE_TTL_SECONDS", "2592000"))  # 30 days
ROLE_CACHE_SCHEMA_VERSION = 1
ROLE_CACHE_MODEL_VERSION = os.getenv("ROLE_CACHE_MODEL_VERSION", "gemini-2.5-flash")


def _new_analysis_record() -> dict:
    now = datetime.utcnow().isoformat() + "Z"
    return {
        "resume_skills": [],
        "job_skills": [],
        "current_role": None,
        "history": [],
        "created_at": now,
        "updated_at": now,
    }


def create_analysis() -> str:
    analysis_id = str(uuid4())
    with ANALYSIS_STORE_LOCK:
        ANALYSIS_STORE[analysis_id] = _new_analysis_record()
    return analysis_id


def get_or_create_analysis(analysis_id: str | None) -> tuple[str, dict]:
    if not analysis_id:
        analysis_id = create_analysis()
    with ANALYSIS_STORE_LOCK:
        record = ANALYSIS_STORE.get(analysis_id)
        if record is None:
            record = _new_analysis_record()
            ANALYSIS_STORE[analysis_id] = record
    return analysis_id, record


def get_analysis(analysis_id: str | None) -> tuple[str, dict | None]:
    if not analysis_id:
        return "", None
    with ANALYSIS_STORE_LOCK:
        return analysis_id, ANALYSIS_STORE.get(analysis_id)


def load_role_db():
    if not os.path.exists(ROLE_DB_FILE):
        return {}

    try:
        with ROLE_DB_LOCK:
            with open(ROLE_DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        if not isinstance(data, dict):
            print("⚠️ role_skills_db.json root is not an object. Starting with empty cache.")
            return {}
        return data
    except json.JSONDecodeError as e:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BASE_DIR, f"role_skills_db.corrupt.{timestamp}.json")
        try:
            shutil.copy2(ROLE_DB_FILE, backup_file)
            print(f"⚠️ Corrupted role DB backed up to: {backup_file}")
        except Exception as backup_error:
            print(f"⚠️ Failed to backup corrupted role DB: {backup_error}")
        print(f"⚠️ Failed to parse role DB ({e}). Starting with empty cache.")
        return {}
    except Exception as e:
        print(f"⚠️ Failed to load role DB ({e}). Starting with empty cache.")
        return {}

def save_role_db(data):
    os.makedirs(os.path.dirname(ROLE_DB_FILE), exist_ok=True)
    payload = data if isinstance(data, dict) else {}

    with ROLE_DB_LOCK:
        fd, temp_path = tempfile.mkstemp(
            prefix="role_skills_db.",
            suffix=".tmp",
            dir=os.path.dirname(ROLE_DB_FILE),
            text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(payload, tmp_file, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(temp_path, ROLE_DB_FILE)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def get_cached_role_skills(role: str) -> tuple[list, bool, bool]:
    """
    Return (skills, is_expired, is_legacy_format) for a cached role entry.
    Legacy format is the old plain list[str] entry.
    """
    entry = ROLE_INTELLIGENCE.get(role)
    if isinstance(entry, list):
        return normalize_skill_list(entry), False, True

    if not isinstance(entry, dict):
        return [], False, False

    skills = normalize_skill_list(entry.get("skills", []))
    expires_at = entry.get("expires_at")
    if isinstance(expires_at, (int, float)) and int(expires_at) < int(datetime.utcnow().timestamp()):
        return [], True, False

    return skills, False, False


def set_cached_role_skills(role: str, skills: list) -> None:
    now_ts = int(datetime.utcnow().timestamp())
    ROLE_INTELLIGENCE[role] = {
        "skills": normalize_skill_list(skills),
        "schema_version": ROLE_CACHE_SCHEMA_VERSION,
        "model_version": ROLE_CACHE_MODEL_VERSION,
        "created_at": now_ts,
        "updated_at": now_ts,
        "expires_at": now_ts + ROLE_CACHE_TTL_SECONDS,
    }


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9\s#+]", " ", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


MIN_SKILLS_CONFIDENCE = 5
MIN_ROLE_SKILLS = 3


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
SKILL_ALIASES.update({
    "scikit-learn": ["scikit learn", "sklearn"],
    "tf-idf": ["tf idf", "tfidf"],
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
    "game development": "game developer",
    "game dev": "game developer",
    "game developer": "game developer",
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
print("🔑 Gemini key loaded:", bool(GEMINI_API_KEY))


GEMINI_ENABLED = bool(GEMINI_API_KEY)
GEMINI_RESUME_MAX_CHARS = int(os.getenv("GEMINI_RESUME_MAX_CHARS", "4500"))
GEMINI_RESUME_MIN_DETERMINISTIC = int(os.getenv("GEMINI_RESUME_MIN_DETERMINISTIC", "8"))
ENABLE_GEMINI_ROLE_FILTER = os.getenv("ENABLE_GEMINI_ROLE_FILTER", "false").lower() == "true"
GEMINI_CONSERVE_MODE = os.getenv("GEMINI_CONSERVE_MODE", "true").lower() == "true"
USE_GEMINI_FOR_RESUME = os.getenv(
    "USE_GEMINI_FOR_RESUME",
    "false" if GEMINI_CONSERVE_MODE else "true"
).lower() == "true"
USE_GEMINI_FOR_KNOWN_ROLE_ASSIST = os.getenv(
    "USE_GEMINI_FOR_KNOWN_ROLE_ASSIST",
    "false"
).lower() == "true"
USE_GEMINI_FOR_UNKNOWN_ROLE = os.getenv(
    "USE_GEMINI_FOR_UNKNOWN_ROLE",
    "true"
).lower() == "true"
GEMINI_MAX_CALLS_PER_HOUR = int(os.getenv("GEMINI_MAX_CALLS_PER_HOUR", "40"))
GEMINI_CALL_LOCK = threading.Lock()
GEMINI_WINDOW_START_TS = int(datetime.utcnow().timestamp())
GEMINI_WINDOW_CALL_COUNT = 0


def can_use_gemini(feature: str = "generic") -> bool:
    """
    Conservative Gemini guard with a rolling hourly budget.
    """
    global GEMINI_WINDOW_START_TS, GEMINI_WINDOW_CALL_COUNT

    if not GEMINI_ENABLED:
        return False

    now_ts = int(datetime.utcnow().timestamp())
    with GEMINI_CALL_LOCK:
        if now_ts - GEMINI_WINDOW_START_TS >= 3600:
            GEMINI_WINDOW_START_TS = now_ts
            GEMINI_WINDOW_CALL_COUNT = 0

        if GEMINI_WINDOW_CALL_COUNT >= GEMINI_MAX_CALLS_PER_HOUR:
            print(f"⚠️ Gemini budget reached ({GEMINI_MAX_CALLS_PER_HOUR}/hour). Skipping: {feature}")
            return False

        GEMINI_WINDOW_CALL_COUNT += 1
        return True

if GEMINI_ENABLED:
    client = genai.Client(api_key=GEMINI_API_KEY)

    print("✅ Gemini AI enabled")
else:
    print("⚠️ Gemini AI disabled — running in offline reasoning mode")





# ===================================
# ðŸ§  ROLE INTELLIGENCE MEMORY
# ===================================
ROLE_INTELLIGENCE = load_role_db()


app = FastAPI()

# =====================================================
# ðŸ§  LOAD EMBEDDING MODEL ON STARTUP (SAFE WAY)
# =====================================================
embedding_model = None
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
SEMANTIC_WEIGHT = 0.75
LEXICAL_WEIGHT = 0.25
EMBEDDING_CACHE_MAX = int(os.getenv("EMBEDDING_CACHE_MAX", "5000"))
EMBEDDING_CACHE_LOCK = threading.Lock()
EMBEDDING_CACHE: Dict[str, Any] = {}

@app.on_event("startup")
def load_embedding_model():
    global embedding_model
    print("🧠 Loading semantic embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"🧠 Embedding model loaded: {EMBEDDING_MODEL_NAME}")


def encode_with_cache(texts: list) -> Any:
    """
    Encode text list with in-memory per-skill embedding cache.
    """
    global embedding_model

    if not texts:
        return embedding_model.encode([], convert_to_tensor=True)

    missing = []
    with EMBEDDING_CACHE_LOCK:
        for text in texts:
            if text not in EMBEDDING_CACHE:
                missing.append(text)

    if missing:
        computed = embedding_model.encode(missing, convert_to_tensor=True)
        if len(missing) == 1 and getattr(computed, "ndim", 0) == 1:
            computed = computed.unsqueeze(0)

        with EMBEDDING_CACHE_LOCK:
            for idx, text in enumerate(missing):
                EMBEDDING_CACHE[text] = computed[idx].detach().cpu()
            while len(EMBEDDING_CACHE) > EMBEDDING_CACHE_MAX:
                EMBEDDING_CACHE.pop(next(iter(EMBEDDING_CACHE)))

    with EMBEDDING_CACHE_LOCK:
        ordered = [EMBEDDING_CACHE[text] for text in texts]
    return torch.stack(ordered)


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

COURSE_LIBRARY = {
    "machine learning": [
        {"title": "Machine Learning Specialization", "platform": "Coursera", "level": "Beginner-Intermediate"},
        {"title": "Intro to Machine Learning", "platform": "Kaggle Learn", "level": "Beginner"},
    ],
    "deep learning": [
        {"title": "Deep Learning Specialization", "platform": "Coursera", "level": "Intermediate"},
        {"title": "Practical Deep Learning for Coders", "platform": "fast.ai", "level": "Intermediate"},
    ],
    "python": [
        {"title": "Python for Everybody", "platform": "Coursera", "level": "Beginner"},
        {"title": "Intermediate Python", "platform": "DataCamp", "level": "Intermediate"},
    ],
    "sql": [
        {"title": "SQL for Data Science", "platform": "Coursera", "level": "Beginner"},
        {"title": "Intro to SQL", "platform": "Kaggle Learn", "level": "Beginner"},
    ],
    "docker": [
        {"title": "Docker Essentials", "platform": "IBM/Coursera", "level": "Beginner"},
        {"title": "Docker for Developers", "platform": "KodeKloud", "level": "Intermediate"},
    ],
    "kubernetes": [
        {"title": "Kubernetes for Beginners", "platform": "KodeKloud", "level": "Beginner"},
        {"title": "CKA Prep Course", "platform": "Linux Foundation", "level": "Advanced"},
    ],
    "api development": [
        {"title": "REST APIs with Flask and Python", "platform": "Udemy", "level": "Intermediate"},
        {"title": "API Design and Fundamentals", "platform": "Postman Academy", "level": "Beginner"},
    ],
    "database systems": [
        {"title": "Database Systems Concepts", "platform": "edX", "level": "Intermediate"},
        {"title": "Relational Databases", "platform": "freeCodeCamp", "level": "Beginner"},
    ],
    "cloud platform": [
        {"title": "AWS Cloud Practitioner", "platform": "AWS Skill Builder", "level": "Beginner"},
        {"title": "Google Cloud Fundamentals", "platform": "Coursera", "level": "Beginner"},
    ],
    "data streaming": [
        {"title": "Apache Kafka Series", "platform": "Udemy", "level": "Intermediate"},
        {"title": "Real-Time Streaming with Spark", "platform": "Databricks Academy", "level": "Advanced"},
    ],
    "apache spark": [
        {"title": "Big Data Analysis with Spark", "platform": "Coursera", "level": "Intermediate"},
        {"title": "Spark Fundamentals", "platform": "Databricks Academy", "level": "Beginner"},
    ],
    "c sharp": [
        {"title": "C# Fundamentals", "platform": "Microsoft Learn", "level": "Beginner"},
        {"title": "C# Intermediate: Classes and OOP", "platform": "Pluralsight", "level": "Intermediate"},
    ],
    "unity engine": [
        {"title": "Unity Essentials Pathway", "platform": "Unity Learn", "level": "Beginner"},
        {"title": "Create with Code", "platform": "Unity Learn", "level": "Beginner-Intermediate"},
    ],
    "unreal engine": [
        {"title": "Unreal Engine 5 Beginner Course", "platform": "Epic Dev Community", "level": "Beginner"},
        {"title": "Unreal C++ Developer", "platform": "Udemy", "level": "Intermediate"},
    ],
    "3d modeling": [
        {"title": "Blender Beginner to Intermediate", "platform": "Blender Studio", "level": "Beginner-Intermediate"},
        {"title": "3D Asset Pipeline Fundamentals", "platform": "CG Cookie", "level": "Intermediate"},
    ],
    "physics engines": [
        {"title": "Game Physics Fundamentals", "platform": "Coursera", "level": "Intermediate"},
        {"title": "Physics for Game Developers", "platform": "Udemy", "level": "Intermediate"},
    ],
    "shader programming": [
        {"title": "Shaders for Game Dev", "platform": "Unity Learn", "level": "Intermediate"},
        {"title": "Intro to Real-Time Shading", "platform": "Udemy", "level": "Intermediate"},
    ],
    "version control": [
        {"title": "Git and GitHub Essentials", "platform": "GitHub Skills", "level": "Beginner"},
        {"title": "Version Control with Git", "platform": "Atlassian University", "level": "Beginner-Intermediate"},
    ],
}

COURSE_KEY_SYNONYMS = {
    "c#": ["c sharp"],
    "c sharp": ["c sharp"],
    "unity": ["unity engine"],
    "unity3d": ["unity engine"],
    "unreal": ["unreal engine"],
    "ue5": ["unreal engine"],
    "ue4": ["unreal engine"],
    "shader": ["shader programming"],
    "shaders": ["shader programming"],
    "graphics programming": ["shader programming"],
    "rendering": ["shader programming"],
    "3d math": ["3d modeling", "physics engines"],
    "game physics": ["physics engines"],
    "physics": ["physics engines"],
    "git": ["version control"],
    "github": ["version control"],
    "api": ["api development"],
    "database": ["database systems"],
    "cloud": ["cloud platform"],
}


def build_recommendations(
    role: str,
    score: int,
    missing_skills: list,
    matched_skills: list
) -> dict:
    """
    Build contextual recommendations and course suggestions using score + gaps.
    """
    role_display = role.title() if role else "Target Role"
    role_normalized = normalize_text(role)
    top_missing = missing_skills[:6]
    top_matched = [m["job_skill"] for m in matched_skills[:3]]

    role_tracks = [
        {
            "keywords": ["ai", "ml", "machine learning", "data scientist", "genai"],
            "project_hint": "model training + evaluation + simple deployment demo",
            "interview_hint": "ML fundamentals, feature engineering, model tradeoffs",
            "course_keys": ["python", "machine learning", "deep learning"],
        },
        {
            "keywords": ["backend", "api", "server", "data engineer"],
            "project_hint": "production-style API with tests, logs, and persistence",
            "interview_hint": "API design, database tradeoffs, scalability basics",
            "course_keys": ["api development", "database systems", "cloud platform"],
        },
        {
            "keywords": ["frontend", "ui", "web", "full stack"],
            "project_hint": "responsive app with performance and accessibility pass",
            "interview_hint": "state management, rendering performance, UX decisions",
            "course_keys": ["api development", "version control", "cloud platform"],
        },
        {
            "keywords": ["devops", "cloud", "sre", "platform"],
            "project_hint": "CI/CD pipeline + IaC + monitoring dashboard",
            "interview_hint": "deployment strategy, reliability, incident handling",
            "course_keys": ["docker", "kubernetes", "cloud platform"],
        },
        {
            "keywords": ["game", "unity", "unreal"],
            "project_hint": "playable prototype with physics and optimization notes",
            "interview_hint": "game loops, rendering, optimization tradeoffs",
            "course_keys": ["unity engine", "shader programming", "physics engines"],
        },
    ]

    track = {
        "project_hint": "role-relevant project with measurable outcomes",
        "interview_hint": "core fundamentals and practical tradeoff discussions",
        "course_keys": ["project-based learning", "interview preparation"],
    }
    for candidate in role_tracks:
        if any(k in role_normalized for k in candidate["keywords"]):
            track = candidate
            break

    gap_count = len(top_missing)
    if score >= 85:
        summary = (
            f"Strong alignment for {role_display}. You are close to interview-ready; close {gap_count} priority gap(s) to maximize conversion."
        )
    elif score >= 65:
        summary = (
            f"Solid baseline for {role_display}. Closing the top {min(3, max(gap_count, 1))} gaps can materially improve readiness."
        )
    else:
        summary = (
            f"Foundational work needed for {role_display}. Prioritize core missing skills before expanding scope."
        )

    focus_areas = []
    if top_missing:
        for idx, skill in enumerate(top_missing[:3], start=1):
            focus_areas.append(
                f"Priority {idx}: Build evidence in '{skill}' with one mini-project, one guided course, and resume bullets showing tools + measurable impact."
            )
    else:
        focus_areas.append("No major skill gaps detected. Focus on project depth, measurable outcomes, and interview consistency.")

    if top_matched:
        focus_areas.append(
            f"Leverage strengths ({', '.join(top_matched)}) to anchor stronger portfolio stories and interview examples."
        )

    action_plan = [
        f"Week 1: Close the highest-priority gap with daily practice and a scoped deliverable.",
        f"Week 2: Build or refine a {track['project_hint']} and publish proof (repo/demo/readme).",
        f"Week 3: Practice interviews around {track['interview_hint']} and update resume with quantified impact.",
    ]

    resume_improvement_tips = [
        f"Customize your resume headline and summary for '{role_display}' and include the top required keywords.",
        "Rewrite project bullets as: action + tech stack + measurable result (latency, accuracy, users, FPS, etc.).",
        "For each key skill, show proof: project link, GitHub repo, demo, or certification.",
        "Keep ATS-friendly formatting: single-column layout, standard section names, consistent skill naming.",
    ]
    if top_missing:
        resume_improvement_tips.append(
            f"Add at least one project bullet that clearly demonstrates '{top_missing[0]}' with outcome metrics."
        )
    if top_matched:
        resume_improvement_tips.append(
            f"Move strengths ({', '.join(top_matched[:2])}) higher in the resume to improve recruiter scan relevance."
        )

    resume_section_feedback = []
    resume_section_feedback.append({
        "section": "Summary",
        "why": f"Recruiters need an immediate role fit for {role_display}.",
        "upgrade": f"Add 2-3 lines with target role, top strengths ({', '.join(top_matched[:2]) if top_matched else 'key technologies'}), and measurable impact."
    })

    if top_missing:
        resume_section_feedback.append({
            "section": "Skills",
            "why": "Missing priority skills are not visible enough in the resume.",
            "upgrade": f"Add/organize skills to highlight: {', '.join(top_missing[:3])}. Keep naming consistent with job requirements."
        })

    resume_section_feedback.append({
        "section": "Projects",
        "why": "Projects are the strongest proof of capability for ATS and recruiters.",
        "upgrade": f"Include at least one project showing {top_missing[0] if top_missing else 'role-critical skills'} with stack + measurable outcome + link."
    })

    resume_section_feedback.append({
        "section": "Experience / Internship",
        "why": "Bullets often describe tasks but not outcomes.",
        "upgrade": "Rewrite bullets using action + tech + result format (e.g., improved accuracy/FPS/latency by X%)."
    })

    resume_section_feedback.append({
        "section": "Certifications / Training",
        "why": "Certifications can quickly close trust gaps for missing fundamentals.",
        "upgrade": f"Add one relevant course/certification for {top_missing[0] if top_missing else 'your target role'} and place it near Projects/Skills."
    })

    def resolve_course_keys_for_skill(skill: str) -> list:
        """
        Resolve best course-library keys for a skill using canonicalization + overlap.
        """
        skill_norm = normalize_skill(normalize_text(skill))
        candidates = []

        if skill_norm in COURSE_LIBRARY:
            candidates.append(skill_norm)

        # Explicit synonym mapping for common market terms.
        candidates.extend(COURSE_KEY_SYNONYMS.get(skill_norm, []))
        for token in skill_norm.split():
            candidates.extend(COURSE_KEY_SYNONYMS.get(token, []))

        # Alias-aware match.
        for canonical, aliases in SKILL_ALIASES.items():
            if skill_norm == canonical or skill_norm in aliases:
                if canonical in COURSE_LIBRARY:
                    candidates.append(canonical)
                for alias in aliases:
                    alias_norm = normalize_skill(normalize_text(alias))
                    if alias_norm in COURSE_LIBRARY:
                        candidates.append(alias_norm)

        # Token overlap fallback.
        skill_tokens = set(skill_norm.split())
        scored = []
        for key in COURSE_LIBRARY.keys():
            key_norm = normalize_skill(normalize_text(key))
            key_tokens = set(key_norm.split())
            overlap = len(skill_tokens.intersection(key_tokens))
            if overlap > 0:
                scored.append((overlap, key))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates.extend([k for _, k in scored[:2]])

        # Contains/contained fallback for phrase variants.
        for key in COURSE_LIBRARY.keys():
            key_norm = normalize_skill(normalize_text(key))
            if key_norm in skill_norm or skill_norm in key_norm:
                candidates.append(key)

        # Unique, preserve order.
        unique = []
        seen = set()
        for key in candidates:
            if key in seen:
                continue
            seen.add(key)
            unique.append(key)
        return unique

    def course_candidates_for_skill(skill: str) -> list:
        keys = resolve_course_keys_for_skill(skill)
        merged = []
        for key in keys:
            merged.extend(COURSE_LIBRARY.get(key, []))
        return merged

    recommended_courses = []
    seen_titles = set()
    seen_platforms = set()
    for skill in top_missing:
        candidates = course_candidates_for_skill(skill)

        for course in candidates:
            title = course["title"]
            if title in seen_titles:
                continue

            # Keep some platform diversity.
            platform = course["platform"]
            if platform in seen_platforms and len(recommended_courses) >= 2:
                continue

            seen_titles.add(title)
            seen_platforms.add(platform)
            recommended_courses.append({
                "for_skill": skill,
                "title": title,
                "platform": platform,
                "level": course["level"],
            })
            if len(recommended_courses) >= 6:
                break
        if len(recommended_courses) >= 6:
            break

    # Role-track fallback if missing-skill mapping is sparse.
    if len(recommended_courses) < 3:
        for key in track.get("course_keys", []):
            for course in COURSE_LIBRARY.get(key, []):
                title = course["title"]
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                recommended_courses.append({
                    "for_skill": key,
                    "title": title,
                    "platform": course["platform"],
                    "level": course["level"],
                })
                if len(recommended_courses) >= 6:
                    break
            if len(recommended_courses) >= 6:
                break

    if not recommended_courses:
        recommended_courses = [
            {"for_skill": "general", "title": "Project-Based Learning Path", "platform": "freeCodeCamp", "level": "Beginner-Intermediate"},
            {"for_skill": "general", "title": "Interview Preparation Track", "platform": "NeetCode / LeetCode", "level": "Intermediate"},
        ]

    return {
        "summary": summary,
        "focus_areas": focus_areas,
        "action_plan": action_plan,
        "priority_gaps": top_missing[:3],
        "strengths": top_matched,
        "resume_improvement_tips": resume_improvement_tips[:6],
        "resume_section_feedback": resume_section_feedback[:5],
        "courses": recommended_courses[:6],
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

BASE_TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "rust",
    "sql", "mysql", "postgresql", "mongodb", "nosql", "sqlite",
    "fastapi", "flask", "django", "spring boot", "node.js", "express",
    "react", "angular", "vue", "html", "css",
    "numpy", "pandas", "scikit-learn", "tensorflow", "pytorch",
    "machine learning", "deep learning", "artificial intelligence",
    "docker", "kubernetes", "git", "github", "linux",
    "aws", "azure", "gcp", "spark", "apache spark", "hadoop", "airflow",
    "etl", "data warehousing", "data streaming", "rest api", "graphql"
}


def build_skill_catalog() -> list:
    """
    Build a deterministic catalog of skills used for exact phrase extraction.
    """
    catalog = set(BASE_TECH_SKILLS)

    for canonical, aliases in SKILL_ALIASES.items():
        catalog.add(canonical)
        catalog.update(aliases)

    for skills in CAPABILITY_SKILLS.values():
        catalog.update(skills)

    return sorted(
        {normalize_skill(normalize_text(s)) for s in catalog if s and len(s.strip()) >= 2},
        key=len,
        reverse=True
    )

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
    if not can_use_gemini("resume_extraction"):
        return []

    prompt = f"""
Extract explicit technical skills from this resume.

Rules:
- Include only concrete tools/technologies (languages, frameworks, libraries, DB, cloud, devops, platforms).
- No soft skills, no responsibilities, no explanations.
- No duplicates.
- Return valid JSON only.

Format:
{{
  "skills": [
    {{ "name": "python", "category": "language" }}
  ]
}}

Resume:
\"\"\"{resume_text[:GEMINI_RESUME_MAX_CHARS]}\"\"\"
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


def catalog_extract_resume_skills(resume_text: str) -> list:
    """
    Phrase-based extractor from a broad technical skill catalog.
    Captures explicit skills even if Gemini omits them.
    """
    text = f" {normalize_text(resume_text)} "
    found = []

    for skill in build_skill_catalog():
        if skill in BLOCKED_SKILLS:
            continue

        # Phrase boundary check to avoid partial-word false positives.
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            found.append(skill)

    return found


def skill_evidence_candidates(skill: str) -> set:
    """
    Build canonical + alias variants to verify a skill is present in resume text.
    """
    normalized_skill = normalize_skill(skill)
    candidates = {normalized_skill}

    # Canonical -> aliases
    aliases = SKILL_ALIASES.get(normalized_skill, [])
    candidates.update(aliases)

    # Alias -> canonical
    for canonical, alias_list in SKILL_ALIASES.items():
        if normalized_skill in alias_list:
            candidates.add(canonical)
            candidates.update(alias_list)

    # Normalize common separators for robust text evidence checks.
    variants = set()
    for c in candidates:
        variants.add(c)
        variants.add(c.replace("-", " "))
        variants.add(c.replace("/", " "))
        variants.add(c.replace(".", " "))
    return {v.strip() for v in variants if v and v.strip()}


def skill_has_text_evidence(skill: str, normalized_resume_text: str) -> bool:
    """
    Verify the skill appears explicitly in resume text.
    """
    text = f" {normalized_resume_text} "
    for candidate in skill_evidence_candidates(skill):
        probe = normalize_text(candidate)
        if not probe:
            continue
        pattern = r"(?<![a-z0-9#+])" + re.escape(probe) + r"(?![a-z0-9#+])"
        if re.search(pattern, text):
            return True
    return False


def post_filter_resume_skills(
    gemini_skills: list,
    keyword_skills: list,
    catalog_skills: list,
    normalized_resume_text: str
) -> list:
    """
    Keep deterministic extractors as trusted evidence and filter Gemini-only noise.
    """
    gemini_clean = normalize_skill_list(gemini_skills)
    keyword_clean = normalize_skill_list(keyword_skills)
    catalog_clean = normalize_skill_list(catalog_skills)

    deterministic = set(keyword_clean) | set(catalog_clean)
    accepted = set(deterministic)

    for skill in gemini_clean:
        if skill in deterministic:
            accepted.add(skill)
            continue
        if skill_has_text_evidence(skill, normalized_resume_text):
            accepted.add(skill)

    return sorted(normalize_skill_list(list(accepted)))


@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    analysis_id: str | None = Form(default=None)
):
    analysis_id, analysis = get_or_create_analysis(analysis_id)

    raw_text = ""

    try:
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                raw_text += page.extract_text() or ""
    except Exception:
        return {"error": "Could not read resume PDF"}
    
    normalized_text = normalize_text(raw_text)


    # Token-efficient extraction: deterministic first, then Gemini only when needed.
    keyword_skills = [normalize_skill(s) for s in keyword_extract_resume_skills(raw_text)]
    catalog_skills = [normalize_skill(s) for s in catalog_extract_resume_skills(raw_text)]
    deterministic_count = len(set(keyword_skills + catalog_skills))

    gemini_skills = []
    if USE_GEMINI_FOR_RESUME and deterministic_count < GEMINI_RESUME_MIN_DETERMINISTIC:
        extracted = gemini_extract_resume_skills(raw_text)
        gemini_skills = [normalize_skill(s["name"]) for s in extracted]

    resume_skills = post_filter_resume_skills(
        gemini_skills,
        keyword_skills,
        catalog_skills,
        normalized_text
    )

    with ANALYSIS_STORE_LOCK:
        analysis["resume_skills"] = resume_skills
        analysis["updated_at"] = datetime.utcnow().isoformat() + "Z"

    return {
        "analysis_id": analysis_id,
        "resume_skills_found": resume_skills,
        "total_skills_detected": len(resume_skills),
        "confidence": "high" if len(resume_skills) >= MIN_SKILLS_CONFIDENCE else "medium"
    }




    



# =====================================================================
#2ï¸âƒ£ DECIDE ON GEMINI USAGE   
#====================================================================


#gemini_infer_role_skills

def should_use_gemini(role: str, skills: list) -> bool:
    """
    Decide whether Gemini should be used.
    """
    if not USE_GEMINI_FOR_KNOWN_ROLE_ASSIST:
        return False

    if not skills:
        return True

    if len(skills) < MIN_SKILLS_CONFIDENCE:
        return True

    if role not in ROLE_CAPABILITIES:
        return True

    return False


def normalize_skill_list(skills: list) -> list:
    """
    Normalize, de-duplicate, and clean a raw skill list.
    """
    cleaned = []
    seen = set()

    for raw in skills or []:
        if not isinstance(raw, str):
            continue
        normalized = normalize_skill(normalize_text(raw))
        if not normalized or normalized in BLOCKED_SKILLS or len(normalized) < 2:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)

    return cleaned


def gemini_filter_relevant_role_skills(role: str, skills: list) -> list:
    """
    Keep only skills that are directly relevant to the given role.
    Gemini must select from provided skills only.
    """
    cleaned_input = normalize_skill_list(skills)
    if not cleaned_input or not can_use_gemini("role_skill_filter"):
        return cleaned_input

    prompt = f"""
You are a technical hiring expert.

ROLE:
{role}

CANDIDATE_SKILLS:
{json.dumps(cleaned_input)}

TASK:
Return ONLY the skills from CANDIDATE_SKILLS that are directly relevant for the ROLE.

RULES:
- Do not add any new skills
- Do not rename skills
- Remove cross-domain or unrelated skills
- Return only valid JSON

FORMAT:
{{
  "skills": []
}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        match = re.search(r"\{[\s\S]*\}", response.text)
        if not match:
            return cleaned_input

        data = json.loads(match.group(0))
        filtered = normalize_skill_list(data.get("skills", []))
        if not filtered:
            return cleaned_input

        # Preserve only original candidate skills even if model output drifts.
        allowed = set(cleaned_input)
        return [s for s in filtered if s in allowed]

    except Exception as e:
        print("⚠️ Gemini role-skill relevance filtering failed:", e)
        return cleaned_input

def gemini_infer_role_skills(role: str) -> dict:
    """
    Ask Gemini ONLY to infer skills for a job role.
    Returns structured JSON or empty result on failure.
    """
    if not USE_GEMINI_FOR_UNKNOWN_ROLE:
        return {}

    if not can_use_gemini("infer_role_skills"):
        return {}

    print("ðŸ¤– Gemini invoked for role:", role)

    prompt = f"""
You are an expert AI system.

Task:
Given a job role, list the core job skills required.

Rules:
- Return ONLY valid JSON
- No explanations
- No markdown
- No extra text
- Skills must be directly relevant to the role
- Avoid unrelated cross-domain skills
- The role can be from any domain (software, marketing, finance, healthcare, education, operations, etc.)
- Prefer concrete tools, platforms, methods, certifications, and domain skills

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

        normalized_skills = normalize_skill_list(skills)
        filtered_skills = (
            gemini_filter_relevant_role_skills(role, normalized_skills)
            if ENABLE_GEMINI_ROLE_FILTER
            else normalized_skills
        )

        result = {
        "role": role,
        "skills": filtered_skills
        }

        # Save only minimally useful results to avoid polluting cache.
        if len(result["skills"]) >= MIN_ROLE_SKILLS:
            set_cached_role_skills(role, result["skills"])
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
    if not can_use_gemini("classify_skill_importance"):
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
    analysis_id = request.get("analysis_id")
    analysis_id, analysis = get_or_create_analysis(analysis_id)

    raw_role = (
        request.get("role")
        or request.get("job_role")
        or request.get("text")
        or ""
    ).lower().strip()
    role = normalize_role(raw_role)
    if not role:
        return {"error": "No role received"}

    with ANALYSIS_STORE_LOCK:
        analysis["current_role"] = role
        analysis["updated_at"] = datetime.utcnow().isoformat() + "Z"

    # Keep in-memory cache synced.
    ROLE_INTELLIGENCE.update(load_role_db())

    # STEP 1: Internal reasoning for known roles (preferred over cache).
    capabilities = ROLE_CAPABILITIES.get(role)
    if capabilities:
        print(f"ðŸ“¥ Role '{role}' handled internally (no Gemini)")
        # STEP 2: Internal skill inference
        skills = []
        for cap in capabilities:
            skills.extend(CAPABILITY_SKILLS.get(cap, []))
        skills = normalize_skill_list(sorted(set(skills)))

        # STEP 3: Optional Gemini assist for weak internal signal
        use_gemini = should_use_gemini(role, skills)
        if use_gemini and GEMINI_ENABLED:
            gemini_result = gemini_infer_role_skills(role)
            gemini_skills = normalize_skill_list(gemini_result.get("skills", []))
            if len(gemini_skills) >= MIN_SKILLS_CONFIDENCE:
                job_skills = sorted(set(gemini_skills))
                with ANALYSIS_STORE_LOCK:
                    analysis["job_skills"] = job_skills
                    analysis["updated_at"] = datetime.utcnow().isoformat() + "Z"
                return {
                    "analysis_id": analysis_id,
                    "role": role,
                    "job_skills_required": job_skills,
                    "confidence": "high",
                    "decision": "gemini_assisted"
                }

        # STEP 4: Final fallback (internal)
        with ANALYSIS_STORE_LOCK:
            analysis["job_skills"] = skills
            analysis["updated_at"] = datetime.utcnow().isoformat() + "Z"
        return {
            "analysis_id": analysis_id,
            "role": role,
            "job_skills_required": skills,
            "confidence": "high" if len(skills) >= MIN_SKILLS_CONFIDENCE else "low",
            "decision": "internal_reasoning"
        }

    # STEP 5: Unknown role -> validate cached Gemini skills before using.
    cached_skills, is_expired, is_legacy = get_cached_role_skills(role)
    if is_expired:
        ROLE_INTELLIGENCE.pop(role, None)
        try:
            save_role_db(dict(ROLE_INTELLIGENCE))
        except Exception as e:
            print("⚠️ Failed to remove expired cache entry:", e)

    if cached_skills:
        validated_cached_skills = (
            gemini_filter_relevant_role_skills(role, cached_skills)
            if (ENABLE_GEMINI_ROLE_FILTER or is_legacy)
            else cached_skills
        )
        if len(validated_cached_skills) >= MIN_ROLE_SKILLS:
            if validated_cached_skills != cached_skills or is_legacy:
                set_cached_role_skills(role, validated_cached_skills)
                try:
                    save_role_db(dict(ROLE_INTELLIGENCE))
                except Exception as e:
                    print("⚠️ Failed to persist validated cache:", e)
            job_skills = sorted(set(validated_cached_skills))
            with ANALYSIS_STORE_LOCK:
                analysis["job_skills"] = job_skills
                analysis["updated_at"] = datetime.utcnow().isoformat() + "Z"
            return {
                "analysis_id": analysis_id,
                "role": role,
                "job_skills_required": job_skills,
                "confidence": "high",
                "decision": "cached_role_validated"
            }

        # Drop weak/irrelevant cache entry so we can refresh via Gemini.
        ROLE_INTELLIGENCE.pop(role, None)
        try:
            save_role_db(dict(ROLE_INTELLIGENCE))
        except Exception as e:
            print("⚠️ Failed to remove stale cache entry:", e)

    # STEP 6: Unknown role -> Gemini primary.
    if USE_GEMINI_FOR_UNKNOWN_ROLE:
        gemini_result = gemini_infer_role_skills(role)
        gemini_skills = normalize_skill_list(gemini_result.get("skills", []))
        if len(gemini_skills) >= MIN_ROLE_SKILLS:
            job_skills = sorted(set(gemini_skills))
            with ANALYSIS_STORE_LOCK:
                analysis["job_skills"] = job_skills
                analysis["updated_at"] = datetime.utcnow().isoformat() + "Z"
            return {
                "analysis_id": analysis_id,
                "role": role,
                "job_skills_required": job_skills,
                "confidence": "high",
                "decision": "gemini_primary"
            }

    # Graceful fallback for any role text when Gemini is unavailable/weak.
    fallback = normalize_skill_list([
        "domain knowledge",
        "communication",
        "problem solving",
        "project management",
        "tools relevant to " + role
    ])
    with ANALYSIS_STORE_LOCK:
        analysis["job_skills"] = fallback
        analysis["updated_at"] = datetime.utcnow().isoformat() + "Z"
    return {
        "analysis_id": analysis_id,
        "role": role,
        "job_skills_required": fallback,
        "confidence": "low",
        "decision": "generic_fallback"
    }


SKILL_FAMILIES = {
    "cloud platform": ["cloud", "aws", "azure", "gcp", "cloud platform"],
    "database systems": ["sql", "mysql", "postgresql", "mongodb", "nosql", "database"],
    "version control": ["git", "github", "gitlab", "bitbucket", "version control"],
    "api development": ["api", "rest", "graphql", "fastapi", "flask", "django"],
    "containerization": ["docker", "kubernetes", "container", "k8s"],
    "etl and pipelines": ["etl", "airflow", "dbt", "data pipeline", "data integration"],
    "data warehousing": ["warehouse", "data warehousing", "snowflake", "bigquery", "redshift"],
    "data streaming": ["streaming", "kafka", "spark streaming", "flink", "rabbitmq"],
    "apache spark": ["apache spark", "spark", "pyspark"],
}


def build_requirement_groups(job_skills: list) -> list:
    """
    Build fair requirement groups so broad requirements are not over-penalized.
    """
    groups = []
    seen_labels = set()

    for raw_skill in job_skills:
        normalized = normalize_text(raw_skill)
        if not normalized:
            continue

        label = normalize_skill(normalized)
        alternatives = {label}
        requirement_type = "core"
        weight = 1.0

        # Map broad requirements into one requirement group with alternatives.
        for family_label, family_terms in SKILL_FAMILIES.items():
            if any(term in normalized for term in family_terms):
                label = family_label
                alternatives.update(normalize_skill(term) for term in family_terms)
                requirement_type = "supporting"
                weight = 0.8
                break

        if label in seen_labels:
            continue

        seen_labels.add(label)
        groups.append({
            "label": label,
            "alternatives": sorted(alternatives),
            "type": requirement_type,
            "weight": weight
        })

    return groups


def lexical_similarity(a: str, b: str) -> float:
    """
    Lexical similarity in [0, 1]. Uses RapidFuzz if available.
    """
    global _RAPIDFUZZ_FUZZ, _RAPIDFUZZ_CHECKED

    if not a or not b:
        return 0.0

    if not _RAPIDFUZZ_CHECKED:
        _RAPIDFUZZ_CHECKED = True
        try:
            module = importlib.import_module("rapidfuzz.fuzz")
            _RAPIDFUZZ_FUZZ = module
        except Exception:
            _RAPIDFUZZ_FUZZ = None

    if _RAPIDFUZZ_FUZZ is not None:
        return float(_RAPIDFUZZ_FUZZ.token_set_ratio(a, b)) / 100.0

    return SequenceMatcher(None, a, b).ratio()


def semantic_match_requirements(resume_skills, requirements, threshold=0.75):
    """
    Match requirements using hybrid semantic + lexical similarity.
    """
    global embedding_model

    matched = []
    missing = []

    if not resume_skills or not requirements:
        return matched, missing, 0

    resume_set = set(resume_skills)
    resume_embeddings = encode_with_cache(resume_skills)

    weighted_hits = 0.0
    total_weight = 0.0

    for req in requirements:
        label = req["label"]
        alternatives = req["alternatives"]
        weight = float(req.get("weight", 1.0))
        total_weight += weight

        # Exact match over any alternative (fair for broad requirement groups).
        exact_hit = next((alt for alt in alternatives if alt in resume_set), None)
        if exact_hit:
            matched.append({
                "job_skill": label,
                "matched_with": exact_hit,
                "similarity": 1.0
            })
            weighted_hits += 1.0 * weight
            continue

        alt_embeddings = encode_with_cache(alternatives)
        sim_matrix = util.cos_sim(alt_embeddings, resume_embeddings)
        best_semantic = float(sim_matrix.max())
        best_flat_idx = int(sim_matrix.argmax())
        best_alt_idx = int(best_flat_idx // len(resume_skills))
        best_resume_idx = int(best_flat_idx % len(resume_skills))
        best_alt_skill = alternatives[best_alt_idx]
        best_resume_skill = resume_skills[best_resume_idx]

        best_lexical = lexical_similarity(best_alt_skill, best_resume_skill)
        hybrid_score = (SEMANTIC_WEIGHT * best_semantic) + (LEXICAL_WEIGHT * best_lexical)

        if hybrid_score >= threshold:
            matched.append({
                "job_skill": label,
                "matched_with": best_resume_skill,
                "similarity": round(hybrid_score, 2)
            })

            if hybrid_score >= 0.90:
                hit_score = 1.0
            elif hybrid_score >= 0.82:
                hit_score = 0.85
            else:
                hit_score = 0.70
            weighted_hits += hit_score * weight
        else:
            missing.append(label)

    score = int((weighted_hits / total_weight) * 100) if total_weight else 0
    return matched, missing, score

       


# =====================================================================
# 3ï¸âƒ£ GET SKILL GAP
# =====================================================================
def compute_ats_score(
    final_score: int,
    normalized_resume: list,
    requirements: list,
    matches: list,
) -> dict:
    """
    Industry-aligned ATS estimate without extra model/API calls.
    Prioritizes exact/core keyword coverage over broad skill volume.
    """
    if not requirements:
        return {
            "ats_score": max(0, min(100, int(final_score))),
            "ats_breakdown": {
                "fit_score": final_score,
                "core_coverage_score": 0,
                "supporting_coverage_score": 0,
                "exact_keyword_score": 0,
                "breadth_score": min(100, int((len(normalized_resume) / 20) * 100)),
                "critical_gap_penalty": 0,
                "requirements_matched": len(matches),
                "requirements_total": 0,
                "industry_band": "insufficient_data"
            }
        }

    match_by_label = {
        m.get("job_skill"): float(m.get("similarity", 0.0))
        for m in matches
        if isinstance(m, dict) and m.get("job_skill")
    }

    core_reqs = [r for r in requirements if r.get("type") == "core"]
    supporting_reqs = [r for r in requirements if r.get("type") != "core"]
    core_total = len(core_reqs)
    supporting_total = len(supporting_reqs)

    core_hits = sum(1 for r in core_reqs if r.get("label") in match_by_label)
    supporting_hits = sum(1 for r in supporting_reqs if r.get("label") in match_by_label)

    core_coverage_score = int((core_hits / core_total) * 100) if core_total else 100
    supporting_coverage_score = int((supporting_hits / supporting_total) * 100) if supporting_total else 100

    # Exact/near-exact keyword signal is a stronger ATS factor than fuzzy proximity.
    exact_hits = sum(1 for sim in match_by_label.values() if sim >= 0.95)
    requirements_total = len(requirements)
    exact_keyword_score = int((exact_hits / requirements_total) * 100) if requirements_total else 0

    # Breadth helps a little, but should not dominate ATS.
    breadth_score = min(100, int((len(normalized_resume) / 20) * 100))

    critical_gap_penalty = 0
    if core_coverage_score < 40:
        critical_gap_penalty = 20
    elif core_coverage_score < 55:
        critical_gap_penalty = 12
    elif core_coverage_score < 70:
        critical_gap_penalty = 6

    ats_score = int(
        (core_coverage_score * 0.45)
        + (supporting_coverage_score * 0.15)
        + (exact_keyword_score * 0.20)
        + (final_score * 0.15)
        + (breadth_score * 0.05)
        - critical_gap_penalty
    )
    ats_score = max(0, min(100, ats_score))

    if ats_score >= 80:
        industry_band = "strong_shortlist_probability"
    elif ats_score >= 65:
        industry_band = "competitive_but_improvable"
    elif ats_score >= 50:
        industry_band = "borderline_screening_risk"
    else:
        industry_band = "high_rejection_risk"

    return {
        "ats_score": ats_score,
        "ats_breakdown": {
            "fit_score": final_score,
            "core_coverage_score": core_coverage_score,
            "supporting_coverage_score": supporting_coverage_score,
            "exact_keyword_score": exact_keyword_score,
            "breadth_score": breadth_score,
            "critical_gap_penalty": critical_gap_penalty,
            "requirements_matched": len(matches),
            "requirements_total": requirements_total,
            "industry_band": industry_band
        }
    }


@app.get("/get-skill-gap")
async def get_skill_gap(analysis_id: str | None = None):
    if not analysis_id:
        with ANALYSIS_STORE_LOCK:
            if len(ANALYSIS_STORE) == 1:
                analysis_id = next(iter(ANALYSIS_STORE))
            else:
                return {"error": "analysis_id is required"}

    _, analysis = get_analysis(analysis_id)
    if not analysis:
        return {"error": f"analysis_id '{analysis_id}' not found"}

    resume_skills = list(analysis.get("resume_skills", []))
    job_skills = list(analysis.get("job_skills", []))
    role = analysis.get("current_role")

    if not resume_skills or not job_skills:
        return {"error": "Upload resume & analyze job first"}

    # 1ï¸âƒ£ Normalize resume skills
    normalized_resume = sorted(set(normalize_skill(s) for s in resume_skills))

    # 2ï¸âƒ£ Capability evaluation (role-based)
    capability_result = evaluate_capabilities(role, normalized_resume)
    capability_score = capability_result["score"]

    # 3ï¸âƒ£ Build grouped requirements (fair scoring, no over-expansion penalties)
    requirements = build_requirement_groups(job_skills)

    # 4ï¸âƒ£ Semantic matching against grouped requirements
    matches, missing, match_score = semantic_match_requirements(
        normalized_resume,
        requirements
    )

    # 5ï¸âƒ£ Blend with capability score only when role has a defined capability model
    has_capability_model = bool(ROLE_CAPABILITIES.get(role))
    if has_capability_model:
        final_score = int((match_score * 0.85) + (capability_score * 0.15))
    else:
        final_score = match_score

    # 6ï¸âƒ£ Extra strengths (positive skills)
    evaluated_labels = {req["label"] for req in requirements}
    extra_strengths = sorted(
        set(normalized_resume) - evaluated_labels
    )
    ats = compute_ats_score(final_score, normalized_resume, requirements, matches)

    result = {
        "analysis_id": analysis_id,
        "resume_skills": resume_skills,
        "job_skills_required": job_skills,
        "semantic_matches": matches,
        "skills_missing": missing,
        "extra_strengths": extra_strengths,
        "match_score": final_score,
        "ats_score": ats["ats_score"],
        "ats_breakdown": ats["ats_breakdown"],
        "recommendations": build_recommendations(
            role or "",
            final_score,
            missing,
            matches
        ),
        "scoring_breakdown": {
            "requirement_match_score": match_score,
            "capability_score": capability_score if has_capability_model else None,
            "capability_model_used": has_capability_model,
            "requirements_evaluated": len(requirements)
        }
    }

    with ANALYSIS_STORE_LOCK:
        history = analysis.setdefault("history", [])
        history.append(result)
        analysis["history"] = history[-10:]
        analysis["updated_at"] = datetime.utcnow().isoformat() + "Z"

    return result

    

    



# =====================================================================
@app.get("/history")
async def get_history(analysis_id: str | None = None):
    if analysis_id:
        _, analysis = get_analysis(analysis_id)
        if not analysis:
            return {"error": f"analysis_id '{analysis_id}' not found"}
        return {"analysis_id": analysis_id, "history": analysis.get("history", [])}

    # Backward-compatible aggregate view when analysis_id is not provided.
    with ANALYSIS_STORE_LOCK:
        merged_history = []
        for aid, record in ANALYSIS_STORE.items():
            for item in record.get("history", []):
                row = dict(item)
                row.setdefault("analysis_id", aid)
                merged_history.append(row)
    return {"history": merged_history[-10:]}
