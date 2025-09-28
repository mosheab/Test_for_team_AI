import os, json, time
from typing import Dict, Any, List
from google import genai

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

schema = {
  "type": "ARRAY",
  "items": {
    "type": "OBJECT",
    "required": ["start_s", "end_s", "title", "summary"],
    "properties": {
      "start_s": {"type": "NUMBER"},
      "end_s": {"type": "NUMBER"},
      "title": {"type": "STRING"},
      "summary": {"type": "STRING"}
    }
  }
}

def llm_summarize_video(filename: str, max_highlights: int = 10) -> Dict[str, Any]:

    def _wait_file_active(file, timeout_s: float = 60.0, poll_s: float = 0.5):
        start = time.time()
        file_id = getattr(file, "name", None) or getattr(file, "id", None)
        if not file_id:
            file_id = file
        while True:
            f = client.files.get(name=file_id)
            state = getattr(f, "state", None)
            if state == "ACTIVE":
                return f
            if state in {"FAILED", "DELETED"}:
                print(f"File {file_id} state={state}, cannot use.")
                return None
            if time.time() - start > timeout_s:
                print(f"File {file_id} not ACTIVE after {timeout_s}s (last state={state}).")
                return None
            time.sleep(poll_s)

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    client = genai.Client(api_key=GOOGLE_API_KEY)
    uploaded = _wait_file_active(client.files.upload(file=filename))
    if not uploaded:
        return []
    prompt = (
        "You analyze a single video and return highlights.\n"
        f"Return ONLY a JSON array with up to {max_highlights} items. "
        "Each item is an object with keys: start_s, end_s, title, summary.\n"
        "Use seconds from 0.0 (e.g., 0.0, 7.5, 22.0, 135.3) for start_s/end_s. Be concise. "
        "Consider visual objects, spoken content, and motion.\n"
        "A highlight should be when something noteworthy happens "
        "(e.g., explosion, crowd cheer, person speaking clearly, high motion, unusual objects).\n"
        "JSON only—no extra text."
    )

    # Schema: top-level ARRAY of highlight OBJECTs
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "required": ["start_s", "end_s", "title", "summary"],
            "properties": {
                "start_s": {"type": "NUMBER"},
                "end_s": {"type": "NUMBER"},
                "title": {"type": "STRING"},
                "summary": {"type": "STRING"}
            }
        }
    }

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, uploaded],  # prompt + video
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.1,
        ),
    )

    # Prefer structured parse
    if getattr(resp, "parsed", None) is not None:
        raw = resp.parsed
    else:
        # Fallback: parse text (handle possible code fences)
        text = (getattr(resp, "text", "") or "").strip()
        if "```" in text:
            block = text.split("```", 2)[1]
            if block.lower().startswith("json"):
                block = block.split("\n", 1)[1] if "\n" in block else ""
            text = block.strip() or text
        raw = json.loads(text)

    # Validate/normalize
    highlights: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        print("Model did not return a JSON array.")
        return highlights

    for item in raw:
        if not isinstance(item, dict):
            continue
        start_s = float(item.get("start_s", 0.0))
        end_s = float(item.get("end_s", max(start_s, 0.0)))
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not (title and summary):
            continue
        try:
            start_s = float(item["start_s"])
            end_s = float(item["end_s"])
        except (KeyError, TypeError, ValueError):
            continue

        if not (0.0 <= start_s < end_s):
            continue

        highlights.append({
            "start_s": start_s,
            "end_s": end_s,
            "title": title,
            "summary": summary,
        })
    return highlights