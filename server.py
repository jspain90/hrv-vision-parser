# server.py
import os, json, shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv, find_dotenv

app = FastAPI()  # <-- must exist at module top level with this exact name

@app.get("/")
def root():
    return {"ok": True}

# Load .env so NOTION_TOKEN/DB_ID and optional UPLOAD_TOKEN are available
_env = find_dotenv()
if _env: load_dotenv(_env)

# Import your existing pipeline pieces.
# IMPORTANT: ensure main.py does NOT run watch_folder() at import-time.
from main import (
    IN_PATH, OUT_PATH, DONE_PATH, FAIL_PATH, VALID_EXTS,
    process_one, resolve_event_time, _unique_target
)

# Optional Notion uploader
NOTION_AVAILABLE = False
try:
    from notion_upload import upload_one
    NOTION_AVAILABLE = True
except Exception:
    NOTION_AVAILABLE = False

UPLOAD_TOKEN = os.getenv("UPLOAD_TOKEN")  # optional simple LAN auth

app = FastAPI()
# Serve our frontend from ./web
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    # Serve the single-page UI
    p = Path("web/index.html")
    if not p.exists():
        return HTMLResponse("<h1>Missing web/index.html</h1>", status_code=500)
    return p.read_text(encoding="utf-8")

@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    x_upload_token: str | None = Header(default=None)  # enforce if set
):
    # Optional shared-secret gate for your LAN
    if UPLOAD_TOKEN and x_upload_token != UPLOAD_TOKEN:
        raise HTTPException(status_code=401, detail="Bad upload token")

    ext = Path(file.filename).suffix.lower()
    if ext not in VALID_EXTS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")

    # Save to screenshots/ with collision-safe name
    dest = _unique_target(Path(IN_PATH), Path(file.filename).name)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Use processing time (your preferred behavior)
    ts = resolve_event_time(str(dest), strategy="now")

    # Run your pipeline immediately (writes JSON to outputs/)
    json_path = process_one(dest.name, event_time=ts)
    if not json_path or not Path(json_path).exists():
        # Move bad upload to failed/
        bad = _unique_target(Path(FAIL_PATH), dest.name)
        shutil.move(str(dest), str(bad))
        raise HTTPException(status_code=500, detail="Processing failed")

    # Move the image to processed/
    done = _unique_target(Path(DONE_PATH), dest.name)
    shutil.move(str(dest), str(done))

    # Read parsed payload to return
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    notion_page_id = None
    if NOTION_AVAILABLE and os.getenv("NOTION_TOKEN") and os.getenv("NOTION_DB_ID"):
        try:
            notion_page_id = upload_one(Path(json_path))
        except Exception as e:
            # Non-fatal; return the error so you can inspect
            notion_page_id = f"ERROR: {e}"

    return {
        "ok": True,
        "file": data.get("file"),
        "timestamp": data.get("timestamp"),
        "parsed": data.get("parsed"),
        "notion_page_id": notion_page_id
    }

if __name__ == "__main__":
    # Run with: uvicorn server:app --host 0.0.0.0 --port 8000
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
