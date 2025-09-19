import os, re, json, time
from PIL import Image, ImageOps
import pytesseract
import os, time, json, shutil
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo   # Python 3.9+. On Windows, tzdata package helps.
from PIL import Image, ExifTags
import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil
from pathlib import Path

# ---- Config ----
IN_PATH   = "screenshots"
OUT_PATH  = "outputs"
DONE_PATH = "processed"
FAIL_PATH = "failed"

LOCAL_TZ = ZoneInfo("America/Chicago")

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(DONE_PATH, exist_ok=True)
os.makedirs(FAIL_PATH, exist_ok=True)

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

class NewImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower() not in VALID_EXTS:
            return
        # wait until fully written
        if not wait_until_stable(p):
            print(f"[WARN] Unstable file skipped: {p.name}")
            return
        # pick a timestamp strategy:
        #  - if you trust “process time as capture time”: use resolve_event_time(..., "now")
        #  - otherwise use "auto" (EXIF -> file -> now)
        ts = resolve_event_time(str(p), strategy="now")
        try:
            json_path = process_one(p.name, event_time=ts)
            if json_path:
                dest = _unique_target(Path(DONE_PATH), p.name)
                shutil.move(str(p), str(dest))
                print(f"[OK] {p.name} → processed/")
            else:
                dest = _unique_target(Path(FAIL_PATH), p.name)
                shutil.move(str(p), str(dest))
                print(f"[FAIL] {p.name} → failed/")
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")
            try:
                dest = _unique_target(Path(FAIL_PATH), p.name)
                shutil.move(str(p), str(dest))
            except Exception as move_err:
                print(f"[WARN] Could not move {p.name} to failed/: {move_err}")

def _unique_target(dst_dir: Path, name: str) -> Path:
    """
    Ensure we don't overwrite files when moving.
    If a file with 'name' exists in dst_dir, append a timestamp.
    """
    target = dst_dir / name
    if target.exists():
        stem, ext = target.stem, target.suffix
        target = target.with_name(f"{stem}_{int(time.time())}{ext}")
    return target

def watch_folder():
    observer = Observer()
    handler = NewImageHandler()
    observer.schedule(handler, IN_PATH, recursive=False)
    observer.start()
    print(f"Watching '{IN_PATH}' for new screenshots… (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def _try_exif_datetime(path: str) -> datetime | None:
    """
    Try to read EXIF 'DateTimeOriginal' (most accurate for photos/screenshots).
    Returns timezone-aware datetime in LOCAL_TZ if present, else None.
    """
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return None
        # Map EXIF tag ids to names
        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        # Common candidates
        for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
            if key in tag_map:
                raw = str(tag_map[key])  # e.g., "2025:09:18 07:42:13"
                try:
                    dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
                    # EXIF is naive (no tz) → interpret as local time of capture
                    return dt.replace(tzinfo=LOCAL_TZ)
                except ValueError:
                    continue
        return None
    except Exception:
        return None

def _file_times(path: str) -> tuple[datetime, datetime]:
    """
    Returns (created, modified) as timezone-aware datetimes in LOCAL_TZ.
    On Windows, st_ctime ~ creation time; on Unix it’s metadata change time.
    """
    st = os.stat(path)
    created = datetime.fromtimestamp(st.st_ctime, tz=LOCAL_TZ)
    modified = datetime.fromtimestamp(st.st_mtime, tz=LOCAL_TZ)
    return created, modified

def resolve_event_time(path: str, strategy: str = "auto") -> dict:
    """
    Returns a dict with 'local_iso' and 'utc_iso' strings.
    strategy:
      - "now"   -> use processing time
      - "file"  -> prefer file creation time
      - "auto"  -> EXIF -> file creation -> now
    """
    now = datetime.now(tz=LOCAL_TZ)
    if strategy == "now":
        event = now
    elif strategy == "file":
        created, _ = _file_times(path)
        event = created
    else:  # "auto"
        event = _try_exif_datetime(path) or _file_times(path)[0] or now

    return {
        "local_iso": event.isoformat(timespec="seconds"),
        "utc_iso": event.astimezone(timezone.utc).isoformat(timespec="seconds"),
    }

def wait_until_stable(p: Path, timeout_s: int = 10, interval_s: float = 0.4) -> bool:
    """
    Returns True if file size stops changing within timeout. Avoids partial reads.
    """
    deadline = time.time() + timeout_s
    last = -1
    while time.time() < deadline:
        sz = p.stat().st_size
        if sz == last and sz > 0:
            return True
        last = sz
        time.sleep(interval_s)
    return False

# LSTM OCR engine; psm 6 = assume a uniform block of text
TESS_CONFIG = r"--oem 3 --psm 4"

# ---- OCR variants (why): white-on-dark UIs often OCR better when inverted ----
def best_ocr_text(img_path: str) -> str:
    img = Image.open(img_path)
    variants = [
        img,
        ImageOps.grayscale(img),
        ImageOps.invert(ImageOps.grayscale(img)),
    ]
    texts = [pytesseract.image_to_string(im, config=TESS_CONFIG) for im in variants]
    # Choose the variant with the most digits (simple, effective heuristic)
    best = max(texts, key=lambda s: len(re.findall(r"\d", s)))
    return best

# ---- Normalization (why): clean recurring OCR artifacts before parsing ----
def normalize_text(s: str) -> str:
    s = s.replace("\u00b2", "2")             # ms² -> ms2
    s = re.sub(r"\bUS\b", " ms", s)          # uppercase US -> ms (rare)
    s = re.sub(r"\bus\b", " ms", s, flags=re.I)  # 'us' (microseconds) misread -> ms
    s = s.replace("PNN5O", "PNN50")          # O/0 confusion (harmless but clean)
    s = re.sub(r"[ \t]+", " ", s)            # collapse spaces
    return s

# ---- Block extraction (why): rely on section anchors, not dropped labels ----
def extract_block(text: str, start_anchor: str, end_anchor: str | None) -> str:
    m1 = re.search(re.escape(start_anchor), text, flags=re.I)
    if not m1:
        return ""
    start = m1.end()
    if end_anchor:
        m2 = re.search(re.escape(end_anchor), text[start:], flags=re.I)
        if m2:
            return text[start:start + m2.start()]
    return text[start:]  # until end

# ---- Typed numeric tokenizer (why): avoid grabbing digits inside words like PNN50 ----
# Replace your iter_numbers and take_sequence with these:

_num_pat = re.compile(r"""
    (?<![A-Za-z0-9])         # not preceded by alnum
    ([-+]?\d+(?:\.\d+)?)     # raw number token
    \s*
    (ms2|ms|hz|%)?           # optional unit token
    (?![A-Za-z0-9])          # not followed by alnum
""", re.IGNORECASE | re.VERBOSE)

def iter_numbers_raw(block: str):
    """
    Yield raw numeric token strings with their raw unit (may be None).
    Do NOT convert to float here; downstream logic may need to repair tokens.
    """
    for m in _num_pat.finditer(block):
        num_str = m.group(1)
        unit = (m.group(2) or "").lower() or None
        yield num_str, unit

def iter_numbers(block: str):
    """
    Yield (value: float, unit_or_None). Repairs the common Hz→'12' artifact
    *before* float conversion so callers like parse_time_domain can use it.
    """
    for num_str, unit in iter_numbers_raw(block):
        # Hz artifact repair: e.g., "0.16412" -> "0.164"
        if (unit == "hz" or unit is None) and num_str.endswith("12") and "." in num_str:
            num_str = num_str[:-2]
        try:
            yield float(num_str), unit
        except ValueError:
            continue

def take_sequence(block: str, expected_units: list[str | None]) -> list[tuple[float, str | None]]:
    """
    Scan tokens in order and pick the next value that matches the expected unit
    (or is plausibly unitless). Special cases:
      • Accept 'ms' when 'ms2' is expected (OCR drops squared).
      • If expected 'hz' and the number token ends with '12' (e.g., 0.16412),
        strip the trailing '12' before float conversion.
    Returns list of (value, normalized_unit_or_None).
    """
    out = []
    i = 0
    for num_str, unit in iter_numbers_raw(block):
        if i >= len(expected_units):
            break

        want = expected_units[i]  # 'ms', 'ms2', 'hz', '%', or None
        normalized_unit = unit

        # Repair common OCR artifact for Hz when unit is missing or wrong:
        if want == "hz":
            # If token looks like 0.12312 (trailing '12' glued on), strip it.
            if num_str.endswith("12") and "." in num_str:
                num_str = num_str[:-2]
            # If unit is present but wrong, we still allow if magnitude is plausible below.

        # Convert after any repairs
        try:
            val = float(num_str)
        except ValueError:
            continue

        if want is None:
            # Unitless slot (e.g., ln(RMSSD), LF/HF); accept any numeric
            pass
        else:
            if unit:
                if unit == want:
                    pass  # exact match
                elif want == "ms2" and unit == "ms":
                    normalized_unit = "ms2"  # accept ms for ms², normalize
                elif want == "hz" and unit != "hz":
                    # mismatched explicit unit for Hz: skip
                    # (we already handled the trailing '12' case above)
                    continue
                else:
                    # explicit mismatched unit -> skip
                    continue
            else:
                # No unit seen: accept only if magnitude is plausible for the expected unit
                if want == "ms"  and not (0 < val <= 3000):  # ms-scale numbers
                    continue
                if want == "ms2" and not (val > 1):          # power usually > 1
                    continue
                if want == "hz"  and not (0 < val < 1.5):    # typical LF/HF peaks
                    continue
                if want == "%"   and not (0 <= val <= 100):
                    continue
                normalized_unit = want  # record the expected unit

        out.append((val, normalized_unit))
        i += 1

    return out


def parse_time_domain(text: str) -> dict:
    block = extract_block(text, "HRV Time-Domain Results", "Frequency-Domain Results")
    # Expect: RMSSD(ms), SDNN(ms), LN_RMSSD(-), PNN50(%), MEAN_RR(ms)
    seq = take_sequence(block, ["ms", "ms", None, "%", "ms"])
    res = {"RMSSD": None, "SDNN": None, "LN_RMSSD": None, "PNN50": None, "MEAN_RR_INTERVAL": None}
    if len(seq) >= 5:
        res["RMSSD"]            = {"value": seq[0][0], "unit": "ms"}
        res["SDNN"]             = {"value": seq[1][0], "unit": "ms"}
        res["LN_RMSSD"]         = {"value": seq[2][0], "unit": None}
        res["PNN50"]            = {"value": seq[3][0], "unit": "%"}
        rr = seq[4][0]
        # Guardrail: RR must be physiologic
        if not (300 <= rr <= 2000):
            # scan for any later plausible ms in this block
            for v, u in iter_numbers(block):
                if (u == "ms" or u is None) and 300 <= v <= 2000:
                    rr = v
        res["MEAN_RR_INTERVAL"] = {"value": rr, "unit": "ms"}
    return res

def parse_frequency_domain(text: str) -> dict:
    block = extract_block(text, "Frequency-Domain Results", None)
    # Expect: Total Power(ms2), LF/HF(-), LF Power(ms2), HF Power(ms2), LF Peak(hz), HF Peak(hz)
    seq = take_sequence(block, ["ms2", None, "ms2", "ms2", "hz", "hz"])
    res = {"TOTAL_POWER": None, "LF_HF_RATIO": None, "LF_POWER": None, "HF_POWER": None, "LF_PEAK": None, "HF_PEAK": None}
    if len(seq) >= 6:
        res["TOTAL_POWER"]  = {"value": seq[0][0], "unit": "ms2"}
        res["LF_HF_RATIO"]  = {"value": seq[1][0], "unit": None}
        res["LF_POWER"]     = {"value": seq[2][0], "unit": "ms2"}
        res["HF_POWER"]     = {"value": seq[3][0], "unit": "ms2"}
        res["LF_PEAK"]      = {"value": seq[4][0], "unit": "hz"}
        res["HF_PEAK"]      = {"value": seq[5][0], "unit": "hz"}
    return res

def process_one(img_name: str, event_time: dict | None = None) -> str | None:
    in_path = os.path.join(IN_PATH, img_name)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing image at {in_path}")

    raw_text = best_ocr_text(in_path)
    text = normalize_text(raw_text)
    td = parse_time_domain(text)
    fd = parse_frequency_domain(text)

    parsed = {
        "RMSSD": td["RMSSD"],
        "SDNN": td["SDNN"],
        "LN_RMSSD": td["LN_RMSSD"],
        "PNN50": td["PNN50"],
        "MEAN_RR_INTERVAL": td["MEAN_RR_INTERVAL"],
        "TOTAL_POWER": fd["TOTAL_POWER"],
        "LF_HF_RATIO": fd["LF_HF_RATIO"],
        "LF_POWER": fd["LF_POWER"],
        "HF_POWER": fd["HF_POWER"],
        "LF_PEAK": fd["LF_PEAK"],
        "HF_PEAK": fd["HF_PEAK"],
    }

    # include capture/processing time metadata
    meta_time = event_time or resolve_event_time(in_path, strategy="auto")

    base = os.path.splitext(os.path.basename(img_name))[0]
    out_name = f"{base}_{int(time.time())}.json"
    out_path = os.path.abspath(os.path.join(OUT_PATH, out_name))

    payload = {
        "file": img_name,
        "timestamp": meta_time,   # <- new
        "parsed": parsed,
        "raw_text": raw_text
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"→ Wrote {out_path}")
        return out_path
    else:
        print(f"[WARN] No output written for {img_name}")
        return None

if __name__ == "__main__":
    # Option A: watch mode (recommended for “take → upload → process immediately”)
    watch_folder()