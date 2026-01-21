import os
import requests

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def gemini_infer_skills(role: str):
    """
    Ask Gemini ONLY when internal logic fails.
    Returns a list of skills.
    """

    if not GEMINI_API_KEY:
        return []

    prompt = f"""
You are an expert job analyst.
Given the job role: "{role}"

Return ONLY a comma-separated list of technical skills required.
No explanation.
"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-1.5-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        res = requests.post(url, json=payload, timeout=10)
        data = res.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        skills = [s.strip().lower() for s in text.split(",")]

        return skills

    except Exception:
        return []
