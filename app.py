import os, csv, base64, io, re
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2

# ✅ ใช้ CPU (กันปัญหา CUDA)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import easyocr

app = Flask(__name__)
app.secret_key = "secret-key-change-me"

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SCAN_DIR = os.path.join(BASE_DIR, "scans")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SCAN_DIR, exist_ok=True)

REG_CSV = os.path.join(DATA_DIR, "registrations.csv")
PASS_CSV = os.path.join(DATA_DIR, "passes.csv")
FAIL_CSV = os.path.join(DATA_DIR, "fails.csv")

REG_HEADERS = ["timestamp", "plate_norm", "plate_raw", "owner", "image_path"]
LOG_HEADERS = ["timestamp", "plate_detected_norm", "plate_detected_raw", "result", "matched_owner", "snapshot_path"]

def ensure_csv(path, headers):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

ensure_csv(REG_CSV, REG_HEADERS)
ensure_csv(PASS_CSV, LOG_HEADERS)
ensure_csv(FAIL_CSV, LOG_HEADERS)

# ---------------------------
# OCR
# ---------------------------
reader = easyocr.Reader(['th', 'en'], gpu=False)
PLATE_KEEP = re.compile(r"[0-9A-Zก-๙]+")

def normalize_plate(s: str) -> str:
    s = (s or "").strip().replace(" ", "").replace("-", "").upper()
    return "".join(PLATE_KEEP.findall(s))

# ---------------------------
# CSV utils
# ---------------------------
def append_csv(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def read_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        reader_obj = csv.DictReader(f)
        for r in reader_obj:
            if not any(r.values()):
                continue
            # ลบ BOM ออกจากคีย์
            r = {k.replace("\ufeff", ""): v for k, v in r.items()}
            rows.append(r)
    print(f"📊 Loaded {os.path.basename(path)}: {len(rows)} rows")
    return rows

# ---------------------------
# Image utils
# ---------------------------
def save_base64_image(data_url, out_dir, prefix):
    if "," in data_url:
        b64 = data_url.split(",", 1)[1]
    else:
        b64 = data_url
    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    out_path = os.path.join(out_dir, filename)
    img.save(out_path, quality=92)

    rel = os.path.relpath(out_path, BASE_DIR).replace("\\", "/")
    return rel

def ocr_plate_from_np(np_img: np.ndarray):
    # ✅ pre-process เพิ่มความแม่นยำ
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    np_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb)  # [(bbox, text, conf), ...]
    texts = [t.strip() for _, t, conf in results if t and conf > 0.40 and len(t.strip()) >= 3]
    raw = " ".join(texts)
    norm = normalize_plate(raw)
    return raw, norm

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/history")
def history():
    regs = read_csv(REG_CSV)[::-1]
    passes = read_csv(PASS_CSV)[::-1]
    fails = read_csv(FAIL_CSV)[::-1]
    return render_template("history.html", regs=regs, passes=passes, fails=fails)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        owner = request.form.get("owner")
        plate_raw = request.form.get("plate")
        file = request.files.get("image")
        if not owner or not plate_raw or not file:
            flash("ข้อมูลไม่ครบ", "error")
            return redirect(url_for("register"))

        safe_name = secure_filename(file.filename or "plate.jpg")
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        save_path = os.path.join(UPLOAD_DIR, name)
        file.save(save_path)
        rel = os.path.relpath(save_path, BASE_DIR).replace("\\", "/")

        append_csv(REG_CSV, [
            datetime.now().isoformat(timespec="seconds"),
            normalize_plate(plate_raw), plate_raw, owner, rel
        ])
        flash("ลงทะเบียนสำเร็จ ✅", "success")
        return redirect(url_for("register"))
    return render_template("register.html")

@app.route("/scan")
def scan():
    return render_template("scan.html")

# ✅ alias ให้ลิงก์เดิมที่ใช้ scan_page ทำงานได้
app.add_url_rule("/scan", endpoint="scan_page", view_func=scan)

@app.route("/api/scan", methods=["POST"])
def api_scan():
    print("📌 /api/scan called")
    data = request.get_json(silent=True) or {}
    data_url = data.get("image")
    if not data_url:
        return jsonify({"ok": False, "error": "no image"}), 400

    try:
        # บันทึก snapshot
        rel_snapshot = save_base64_image(data_url, SCAN_DIR, "scan")
        abs_snapshot = os.path.join(BASE_DIR, rel_snapshot)
        print("✅ Saved snapshot:", rel_snapshot)

        # เปิดภาพสำหรับ OCR
        pil_img = Image.open(abs_snapshot).convert("RGB")
        np_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("❌ Error processing image:", e)
        return jsonify({"ok": False, "error": str(e)}), 500

    detected_raw, detected_norm = ocr_plate_from_np(np_img)
    print("🔍 OCR:", detected_raw, "=>", detected_norm)

    regs = read_csv(REG_CSV)
    plate2owner = {r["plate_norm"]: r["owner"] for r in regs}
    matched_owner = plate2owner.get(detected_norm, "")
    now = datetime.now().isoformat(timespec="seconds")

    if matched_owner:
        result = "PASS"
        append_csv(PASS_CSV, [now, detected_norm, detected_raw, result, matched_owner, rel_snapshot])
        print("✅ PASS logged")
    else:
        result = "FAIL"
        append_csv(FAIL_CSV, [now, detected_norm, detected_raw, result, "", rel_snapshot])
        print("✅ FAIL logged")

    return jsonify({
        "ok": True,
        "result": result,
        "detected_raw": detected_raw,
        "detected_norm": detected_norm,
        "matched_owner": matched_owner,
        "snapshot_url": "/" + rel_snapshot
    })

# ✅ เสิร์ฟรูปที่อัปโหลด/สแกน
@app.route("/uploads/<path:name>")
def uploads(name):
    return send_from_directory(UPLOAD_DIR, name)

@app.route("/scans/<path:name>")
def scans(name):
    return send_from_directory(SCAN_DIR, name)

if __name__ == "__main__":
    # รัน: python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)