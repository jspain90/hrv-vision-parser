# notion_upload.py
import os, json, time, shutil
from pathlib import Path
from typing import Any, Dict, Optional
import requests
import os
from dotenv import load_dotenv, find_dotenv

OUTPUTS_DIR     = Path("outputs")
UPLOADED_DIR    = Path("uploaded")
FAILED_DIR      = Path("upload_failed")
for d in (UPLOADED_DIR, FAILED_DIR):
    d.mkdir(exist_ok=True)

_env_path = find_dotenv()  # walks up directories to find .env
if _env_path:
    load_dotenv(_env_path)
else:
    # Fall back to current working directory; harmless if no .env present
    load_dotenv()

def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"Missing env var {name}. "
            f"Looked for .env at: {_env_path or '<cwd>/.env'}. "
            "Set it there or export it in your shell."
        )
    return v

NOTION_TOKEN = _require_env("NOTION_TOKEN")
NOTION_DB_ID = _require_env("NOTION_DB_ID")

if not NOTION_TOKEN or not NOTION_DB_ID:
    raise SystemExit("Set NOTION_TOKEN and NOTION_DB_ID in your environment (or .env).")

SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",  # stable; newer works too
    "Content-Type": "application/json",
})

def _num(x: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract numeric value from {'value': float, 'unit': ...} or return None."""
    if not x:
        return None
    return x.get("value")

def build_properties(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map our JSON to Notion DB properties. Keep names EXACTLY as in your DB.
    Missing values -> omit the property (simplest way to avoid type errors).
    """
    parsed = payload.get("parsed", {})
    ts     = payload.get("timestamp", {})
    props  = {}

    # Title (String). Notion requires one title prop; name yours 'Source File'.
    source_name = payload.get("file") or "HRV Reading"
    props["Source File"] = {"title": [{"text": {"content": source_name}}]}

    # Date (ISO). We use utc_iso to be unambiguous.
    if ts.get("utc_iso"):
        props["Reading Time"] = {"date": {"start": ts["utc_iso"]}}

    # Numbers. Only set when value is present to avoid null-type errors.
    def maybe(prop_name: str, value: Optional[float]):
        if value is not None:
            props[prop_name] = {"number": float(value)}

    maybe("RMSSD",            _num(parsed.get("RMSSD")))
    maybe("SDNN",             _num(parsed.get("SDNN")))
    maybe("LN_RMSSD",         _num(parsed.get("LN_RMSSD")))
    maybe("PNN50",            _num(parsed.get("PNN50")))
    maybe("MEAN_RR_INTERVAL", _num(parsed.get("MEAN_RR_INTERVAL")))
    maybe("TOTAL_POWER",      _num(parsed.get("TOTAL_POWER")))
    maybe("LF_HF_RATIO",      _num(parsed.get("LF_HF_RATIO")))
    maybe("LF_POWER",         _num(parsed.get("LF_POWER")))
    maybe("HF_POWER",         _num(parsed.get("HF_POWER")))
    maybe("LF_PEAK",          _num(parsed.get("LF_PEAK")))
    maybe("HF_PEAK",          _num(parsed.get("HF_PEAK")))

    return props

def notion_create_page(properties: Dict[str, Any], children: Optional[list] = None) -> Dict[str, Any]:
    """
    POST /v1/pages with basic retry on 429/5xx.
    """
    body = {"parent": {"database_id": NOTION_DB_ID}, "properties": properties}
    if children:
        body["children"] = children

    backoff = 1.0
    for attempt in range(5):
        resp = SESSION.post("https://api.notion.com/v1/pages", data=json.dumps(body))
        if resp.status_code in (200, 201):
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            # Respect Retry-After if provided
            ra = float(resp.headers.get("Retry-After", backoff))
            time.sleep(ra)
            backoff = min(backoff * 2, 10)
            continue
        # Hard failure: show the server message to debug property name/type mismatches
        raise RuntimeError(f"Notion error {resp.status_code}: {resp.text[:400]}")

    raise RuntimeError("Notion create page: exhausted retries")

def upload_one(json_path: Path) -> Optional[str]:
    """
    Load a single outputs/*.json and create a Notion page. Return page id on success.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    props = build_properties(data)

    # Optional: add a raw_text collapsible block for debugging (toggle on/off).
    children = [
        {
            "toggle": {
                "rich_text": [{"text": {"content": "Raw OCR"}}],
                "children": [
                    {"paragraph": {"rich_text": [{"text": {"content": data.get("raw_text","")[:1900]}}]}}
                ],
            }
        }
    ]

    res = notion_create_page(props, children=children)
    return res.get("id")

def upload_all():
    files = sorted([p for p in OUTPUTS_DIR.glob("*.json") if p.is_file()])
    if not files:
        print("No JSON outputs to upload.")
        return

    ok, fail = 0, 0
    for p in files:
        try:
            page_id = upload_one(p)
            if page_id:
                dest = UPLOADED_DIR / p.name
                if dest.exists():
                    dest = dest.with_stem(dest.stem + f"_{int(time.time())}")
                shutil.move(str(p), str(dest))
                print(f"[OK] {p.name} → Notion page {page_id}")
                ok += 1
            else:
                raise RuntimeError("No page id returned")
        except Exception as e:
            print(f"[FAIL] {p.name}: {e}")
            dest = FAILED_DIR / p.name
            if dest.exists():
                dest = dest.with_stem(dest.stem + f"_{int(time.time())}")
            shutil.move(str(p), str(dest))
            fail += 1

    print(f"\nUpload summary: ok={ok}, failed={fail}. "
          f"Uploaded JSONs archived in '{UPLOADED_DIR}', failures in '{FAILED_DIR}'.")
    
if __name__ == "__main__":
    upload_all()
