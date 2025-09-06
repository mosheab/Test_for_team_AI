import os, json
from typing import Dict, Any
from google import genai

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def llm_summarize_scene(scene_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    client = genai.Client(api_key=GOOGLE_API_KEY)
    sys_prompt = (
        "You analyze scenes from a video and decide if a scene is a highlight. "
        "Return ONLY a compact JSON object with keys: is_highlight (true/false), title, summary. "
        "Consider objects, speech, motion, and timestamps."
    )
    user_payload = json.dumps(scene_payload, ensure_ascii=False)
    prompt = sys_prompt + "\n\nSCENE JSON:\n" + user_payload
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = getattr(resp, "text", "").strip()
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.lower().startswith("json"):
                text = text.split("\n", 1)[1]
        data = json.loads(text)
        return {
            "is_highlight": bool(data.get("is_highlight")),
            "title": str(data.get("title") or ""),
            "summary": str(data.get("summary") or ""),
        }
    except Exception:
        return {"is_highlight": False, "title": "", "summary": ""}
