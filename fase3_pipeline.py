"""
Pipeline Integration (Computer Vision + Soft Computing)
"""

import sys
import os
import time
import math
import random
import urllib.request
from collections import Counter
import numpy as np
import cv2

# MediaPipe Tasks API (berlaku untuk mediapipe >= 0.10.x)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult

# Import mesin fuzzy dari Fase 2 (harus ada di folder yang sama)
try:
    from fase2_fuzzy_engine import (
        build_fuzzy_variables,
        build_rules,
        build_control_system,
        compute_engagement,
        class_to_emosi_value,
        CNN_CLASS_TO_EMOSI,
    )
    FUZZY_AVAILABLE = True
except ImportError:
    print("[WARN] fase2_fuzzy_engine.py tidak ditemukan. Fuzzy score = fallback.")
    FUZZY_AVAILABLE = False

# Configuration
CAMERA_ID        = 0
FRAME_W, FRAME_H = 1280, 720
FUZZY_INTERVAL   = 1.0          # detik antar komputasi fuzzy
EAR_SMOOTH_N     = 5
POSE_SMOOTH_N    = 5
EMOTION_VOTE_N   = 20           # frame untuk majority-vote stabilisasi emosi

# Kalibrasi EAR personal
CALIB_SECONDS    = 3            # durasi kalibrasi (detik)
CALIB_TARGET_EAR = 0.32        # nilai EAR "waspada" target setelah normalisasi
EAR_DROWSY_RATIO = 0.60        # jika EAR < 60% baseline → dianggap mengantuk
EAR_TIRED_RATIO  = 0.78        # jika EAR < 78% baseline → dianggap lelah

# URL model MediaPipe Face Landmarker (diunduh otomatis jika belum ada)
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = "face_landmarker.task"

# Warna overlay (BGR)
COLOR_GREEN  = (0, 220, 100)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED    = (0, 60, 230)
COLOR_WHITE  = (255, 255, 255)
COLOR_DARK   = (20, 20, 20)
COLOR_CYAN   = (255, 220, 0)

# MediaPipe Tasks API Landmark Indices
# 6 titik per mata untuk EAR (posisi p1..p6 searah jarum jam)
# Indeks sama dengan Face Mesh lama karena pakai 478-landmark model
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# Titik referensi 3D wajah generik untuk solvePnP (satuan mm)
MODEL_3D_POINTS = np.array([
    (0.0,    0.0,    0.0),    # Ujung hidung  (lm 1)
    (0.0,  -330.0,  -65.0),   # Dagu          (lm 152)
    (-225.0, 170.0, -135.0),  # Mata kiri     (lm 263)
    (225.0,  170.0, -135.0),  # Mata kanan    (lm 33)
    (-150.0,-150.0, -125.0),  # Mulut kiri    (lm 287)
    (150.0, -150.0, -125.0),  # Mulut kanan   (lm 57)
], dtype=np.float64)
POSE_LM_IDX = [1, 152, 263, 33, 287, 57]

EMOTION_CLASSES = ["Negatif", "Bingung", "Netral", "Positif"]

# Download Model

def ensure_model(path: str, url: str):
    if os.path.exists(path):
        return
    print(f"[INFO] Mengunduh model Face Landmarker (~30 MB)...")
    print(f"       {url}")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"[INFO] Model disimpan: {path}")
    except Exception as e:
        sys.exit(f"[ERROR] Gagal mengunduh model: {e}\n"
                 f"Unduh manual dari:\n  {url}\nLalu simpan sebagai '{path}'")

# EAR Calculation

def _dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def compute_ear(face_landmarks, eye_idx, img_w, img_h):
    pts = []
    for idx in eye_idx:
        lm = face_landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))

    p1, p2, p3, p4, p5, p6 = pts
    denom = 2.0 * _dist(p1, p4)
    if denom < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / denom


# Personal EAR Calibration

def calibrate_ear(cap, landmarker) -> float:
    print("[KALIB] Tekan SPASI saat posisi Anda sudah netral/normal untuk submit baseline.")
    print("[KALIB] Tekan 'q' untuk lewati kalibrasi (pakai default 0.30).")

    # Rolling buffer — simpan 30 frame terakhir (~1 detik di 30fps)
    BUFFER_SIZE = 30
    ear_buffer  = []
    submitted   = False
    baseline    = 0.30  # default fallback

    while not submitted:
        ret, frame = cap.read()
        if not ret:
            continue

        frame        = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        # --- Latar gelap semi-transparan ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, img_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

        # --- Judul ---
        cv2.putText(frame, "KALIBRASI KONDISI NETRAL",
                    (img_w//2 - 220, img_h//2 - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_CYAN, 2)
        cv2.line(frame,
                 (img_w//2 - 220, img_h//2 - 90),
                 (img_w//2 + 220, img_h//2 - 90),
                 COLOR_CYAN, 1)

        # --- Instruksi ---
        instructions = [
            "1. Duduk dengan posisi normal di depan kamera",
            "2. Tatap layar seperti biasa saat belajar",
            "3. Perhatikan nilai EAR di bawah hingga stabil",
            "4. Tekan  SPASI  untuk mengunci baseline Anda",
        ]
        for i, txt in enumerate(instructions):
            cv2.putText(frame, txt,
                        (img_w//2 - 280, img_h//2 - 55 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)

        # --- Deteksi wajah & hitung EAR ---
        ear_now = None
        if result.face_landmarks:
            lm    = result.face_landmarks[0]
            ear_l = compute_ear(lm, LEFT_EYE_IDX,  img_w, img_h)
            ear_r = compute_ear(lm, RIGHT_EYE_IDX, img_w, img_h)
            ear_now = (ear_l + ear_r) / 2.0

            if ear_now > 0.05:
                ear_buffer.append(ear_now)
                if len(ear_buffer) > BUFFER_SIZE:
                    ear_buffer.pop(0)

            # --- Panel EAR live ---
            ear_median = float(np.median(ear_buffer)) if ear_buffer else 0.0
            ear_std    = float(np.std(ear_buffer))    if len(ear_buffer) > 2 else 0.0
            stable     = ear_std < 0.012              # stabil jika std kecil

            # Warna panel: hijau = stabil, kuning = belum stabil
            panel_color = COLOR_GREEN if stable else COLOR_ORANGE
            stability_txt = "STABIL — siap submit!" if stable else "Tunggu hingga stabil..."

            cv2.rectangle(frame,
                          (img_w//2 - 200, img_h//2 + 60),
                          (img_w//2 + 200, img_h//2 + 160), (30, 30, 30), -1)
            cv2.rectangle(frame,
                          (img_w//2 - 200, img_h//2 + 60),
                          (img_w//2 + 200, img_h//2 + 160), panel_color, 2)

            cv2.putText(frame, f"EAR saat ini : {ear_now:.4f}",
                        (img_w//2 - 185, img_h//2 + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
            cv2.putText(frame, f"EAR median   : {ear_median:.4f}  (std: {ear_std:.4f})",
                        (img_w//2 - 185, img_h//2 + 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
            cv2.putText(frame, stability_txt,
                        (img_w//2 - 185, img_h//2 + 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, panel_color, 1)

        else:
            # Tidak ada wajah
            cv2.putText(frame, "⚠ Wajah tidak terdeteksi — posisikan kembali",
                        (img_w//2 - 280, img_h//2 + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2)

        # --- Petunjuk tombol di bawah ---
        cv2.putText(frame, "[ SPASI ] Submit baseline    [ Q ] Lewati",
                    (img_w//2 - 220, img_h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CYAN, 1)

        cv2.imshow("Smart Engagement Tracker — Fase 3", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):   # SPASI → submit
            if len(ear_buffer) >= 10:
                baseline    = float(np.median(ear_buffer))
                submitted   = True
                print(f"[KALIB] ✓ Baseline dikunci: EAR = {baseline:.4f}  "
                      f"(dari {len(ear_buffer)} sampel)")
                # Tampilkan konfirmasi 1.5 detik
                confirm_t = time.time()
                while (time.time() - confirm_t) < 1.5:
                    ret2, frame2 = cap.read()
                    if not ret2:
                        continue
                    frame2 = cv2.flip(frame2, 1)
                    overlay2 = frame2.copy()
                    cv2.rectangle(overlay2, (0, 0), (frame2.shape[1], frame2.shape[0]),
                                  (0, 0, 0), -1)
                    cv2.addWeighted(overlay2, 0.6, frame2, 0.4, 0, frame2)
                    cv2.putText(frame2, f"Baseline tersimpan: {baseline:.4f}",
                                (frame2.shape[1]//2 - 220, frame2.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_GREEN, 2)
                    cv2.putText(frame2, "Memulai tracking...",
                                (frame2.shape[1]//2 - 140, frame2.shape[0]//2 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 1)
                    cv2.imshow("Smart Engagement Tracker — Fase 3", frame2)
                    cv2.waitKey(1)
            else:
                print("[KALIB] Belum cukup sampel, posisikan wajah ke kamera dulu.")

        elif key == ord("q"):  # Lewati kalibrasi
            print(f"[KALIB] Kalibrasi dilewati, pakai default baseline = {baseline:.2f}")
            break

    return baseline


def normalize_ear(ear_raw: float, baseline: float) -> float:
    if baseline < 0.01:
        return float(np.clip(ear_raw, 0.0, 0.50))
    scale    = CALIB_TARGET_EAR / baseline
    ear_norm = ear_raw * scale
    return float(np.clip(ear_norm, 0.0, 0.50))

# Head Pose Estimation

def compute_head_yaw(face_landmarks, img_w, img_h):
    # Landmark yang dipakai:
    #   1   = ujung hidung
    #   33  = sudut luar mata kanan (tampilan mirror = kiri di layar)
    #   263 = sudut luar mata kiri  (tampilan mirror = kanan di layar)
    nose  = face_landmarks[1]
    l_eye = face_landmarks[33]
    r_eye = face_landmarks[263]

    nx  = nose.x  * img_w
    lx  = l_eye.x * img_w
    rx  = r_eye.x * img_w

    eye_width = abs(rx - lx)
    if eye_width < 1e-3:
        return 0.0

    # Deviasi relatif: 0 = lurus, ±0.5 = menoleh ~45°
    mid_x     = (lx + rx) / 2.0
    deviation = abs(nx - mid_x) / eye_width

    # Konversi ke derajat; secara empiris deviation ~0.40 ≈ 45°
    # Clamp ke [0, 90]
    yaw_deg = deviation * (90.0 / 0.40)
    return min(yaw_deg, 90.0)

# CNN Placeholder

def dummy_cnn_predict(face_roi_bgr):
    raw   = [random.random() for _ in EMOTION_CLASSES]
    total = sum(raw)
    return {cls: float(r/total) for cls, r in zip(EMOTION_CLASSES, raw)}

def top_emotion(prob_dict):
    return max(prob_dict, key=prob_dict.get)

# Fuzzy Engagement Wrapper

def calculate_fuzzy_engagement(simulation, ear, pose_deg, emotion_class):
    if not FUZZY_AVAILABLE or simulation is None:
        score = 50.0
        if ear < EAR_THRESHOLD: score -= 30
        if pose_deg > 25:       score -= 20
        if emotion_class == "Positif": score += 20
        elif emotion_class == "Negatif": score -= 10
        return float(max(0.0, min(100.0, score)))

    emosi_val = class_to_emosi_value(emotion_class)
    return compute_engagement(simulation, ear, pose_deg, emosi_val, verbose=False)

# Rendering Overlay

def _eng_color(s):
    return COLOR_GREEN if s >= 65 else (COLOR_ORANGE if s >= 35 else COLOR_RED)

def _eng_label(s):
    return "TINGGI" if s >= 65 else ("SEDANG" if s >= 35 else "RENDAH")

def draw_eye_pts(frame, face_lm, eye_idx, img_w, img_h, color):
    pts = [(int(face_lm[i].x * img_w), int(face_lm[i].y * img_h)) for i in eye_idx]
    for i in range(len(pts)):
        cv2.line(frame, pts[i], pts[(i+1) % len(pts)], color, 1)
    for pt in pts:
        cv2.circle(frame, pt, 2, color, -1)

def draw_face_bbox(frame, face_lm, img_w, img_h):
    xs = [int(lm.x * img_w) for lm in face_lm]
    ys = [int(lm.y * img_h) for lm in face_lm]
    x1 = max(min(xs) - 10, 0); y1 = max(min(ys) - 10, 0)
    x2 = min(max(xs) + 10, img_w); y2 = min(max(ys) + 10, img_h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_CYAN, 2)
    return x1, y1, x2, y2

def draw_hud(frame, ear_raw, ear_norm, baseline, pose_deg, emotion, score, fps):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (315, 235), COLOR_DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "SMART ENGAGEMENT TRACKER",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1)
    cv2.line(frame, (16, 36), (308, 36), COLOR_CYAN, 1)

    # Tentukan status EAR berdasarkan rasio terhadap baseline
    ear_ratio   = ear_raw / max(baseline, 0.01)
    ear_status  = "WASPADA" if ear_ratio > EAR_TIRED_RATIO else \
                  ("LELAH" if ear_ratio > EAR_DROWSY_RATIO else "MENGANTUK!")
    ear_color   = COLOR_GREEN if ear_ratio > EAR_TIRED_RATIO else \
                  (COLOR_ORANGE if ear_ratio > EAR_DROWSY_RATIO else COLOR_RED)

    rows = [
        (f"FPS     : {fps:>5.1f}",                    COLOR_WHITE),
        (f"EAR raw : {ear_raw:>5.3f} (base:{baseline:.3f})", COLOR_WHITE),
        (f"EAR norm: {ear_norm:>5.3f}  [{ear_status}]",     ear_color),
        (f"Pose    : {pose_deg:>5.1f} deg",               COLOR_WHITE),
        (f"Emosi   : {emotion}",                          COLOR_WHITE),
    ]
    y = 55
    for text, color in rows:
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        y += 20

    cv2.line(frame, (16, y), (308, y), COLOR_CYAN, 1)
    y += 18
    eng_c = _eng_color(score)
    cv2.putText(frame, f"ENGAGE  : {score:>5.1f}/100  {_eng_label(score)}",
                (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, eng_c, 1)
    y += 18
    bar_w = int((score / 100.0) * 290)
    cv2.rectangle(frame, (16, y), (306, y+10), (60, 60, 60), -1)
    cv2.rectangle(frame, (16, y), (16 + bar_w, y+10), eng_c, -1)

def draw_emotion_probs(frame, probs):
    h, w = frame.shape[:2]
    x0, y0 = w - 185, 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0-5, y0-5), (w-8, y0 + len(probs)*28 + 10), COLOR_DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    PCOLORS = {"Positif": COLOR_GREEN, "Netral": COLOR_CYAN,
               "Bingung": COLOR_ORANGE, "Negatif": COLOR_RED}
    y = y0 + 15
    for cls in EMOTION_CLASSES:
        p = probs.get(cls, 0.0)
        c = PCOLORS.get(cls, COLOR_WHITE)
        cv2.putText(frame, f"{cls[:3]}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        cv2.rectangle(frame, (x0+35, y-10), (x0+35+int(p*120), y), c, -1)
        cv2.putText(frame, f"{p:.2f}", (x0+155, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1)
        y += 28

# Init MediaPipe Tasks

def init_face_landmarker(model_path: str):
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,   # mode per-frame (sinkron)
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    print("[INFO] MediaPipe FaceLandmarker (Tasks API) siap.")
    return landmarker

# Init Fuzzy Engine

def init_fuzzy_engine():
    if not FUZZY_AVAILABLE:
        return None
    print("[INFO] Menginisialisasi Fuzzy Engine Fase 2...")
    ear_var, pose_var, emosi_var, engage_var = build_fuzzy_variables()
    rules = build_rules(ear_var, pose_var, emosi_var, engage_var)
    _, simulation = build_control_system(rules)
    print("[INFO] Fuzzy Engine siap.")
    return simulation

# Main Loop

def main():
    print("=" * 60)
    print("  FASE 3: PIPELINE INTEGRASI REAL-TIME")
    print("  Smart Learning Engagement Tracker")
    print(f"  MediaPipe v{mp.__version__}")
    print("=" * 60)
    print("[INFO] Tekan 'q' keluar | 's' snapshot | 'r' kalibrasi ulang")

    # 1. Download model jika belum ada
    ensure_model(MODEL_PATH, MODEL_URL)

    # 2. Inisialisasi Fuzzy Engine
    simulation = init_fuzzy_engine()

    # 3. Inisialisasi MediaPipe Face Landmarker (Tasks API)
    landmarker = init_face_landmarker(MODEL_PATH)

    # 4. Setup kamera
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Kamera ID={CAMERA_ID} tidak dapat dibuka.")

    # 5. Kalibrasi personal EAR
    ear_baseline = calibrate_ear(cap, landmarker)
    print(f"[INFO] Baseline EAR: {ear_baseline:.4f}  "
          f"| Threshold mengantuk < {ear_baseline * EAR_DROWSY_RATIO:.3f}  "
          f"| Threshold lelah < {ear_baseline * EAR_TIRED_RATIO:.3f}")

    # State
    ear_buf, pose_buf      = [], []
    emotion_votes          = []   # buffer majority-vote untuk stabilisasi emosi
    last_fuzzy_t = 0.0
    fuzzy_score  = 50.0
    emotion_cls  = "Netral"
    probs        = {c: 0.25 for c in EMOTION_CLASSES}
    fps          = 0.0
    prev_t       = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame    = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]
        now      = time.time()
        fps      = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
        prev_t   = now

        # Konversi ke MediaPipe Image object (Tasks API memakai mp.Image)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Deteksi landmark
        result: FaceLandmarkerResult = landmarker.detect(mp_image)

        if result.face_landmarks:
            # face_landmarks[0] = list NormalizedLandmark wajah pertama
            lm = result.face_landmarks[0]

            # (a) Bounding box
            x1, y1, x2, y2 = draw_face_bbox(frame, lm, img_w, img_h)

            # (b) EAR kedua mata
            ear_l   = compute_ear(lm, LEFT_EYE_IDX,  img_w, img_h)
            ear_r   = compute_ear(lm, RIGHT_EYE_IDX, img_w, img_h)
            ear_raw = (ear_l + ear_r) / 2.0
            ear_buf.append(ear_raw)
            if len(ear_buf) > EAR_SMOOTH_N: ear_buf.pop(0)
            ear_raw_s = float(np.mean(ear_buf))

            # Normalisasi EAR ke skala yang konsisten dengan MF fuzzy
            ear_s = normalize_ear(ear_raw_s, ear_baseline)

            ear_ratio = ear_raw_s / max(ear_baseline, 0.01)
            eye_color = COLOR_GREEN  if ear_ratio > EAR_TIRED_RATIO else \
                        (COLOR_ORANGE if ear_ratio > EAR_DROWSY_RATIO else COLOR_RED)
            draw_eye_pts(frame, lm, LEFT_EYE_IDX,  img_w, img_h, eye_color)
            draw_eye_pts(frame, lm, RIGHT_EYE_IDX, img_w, img_h, eye_color)

            # (c) Head pose Yaw
            yaw = compute_head_yaw(lm, img_w, img_h)
            pose_buf.append(yaw)
            if len(pose_buf) > POSE_SMOOTH_N: pose_buf.pop(0)
            yaw_s = float(np.mean(pose_buf))

            # (d) Crop wajah → CNN dummy
            roi = frame[max(y1,0):y2, max(x1,0):x2]
            if roi.size > 0:
                probs        = dummy_cnn_predict(roi)
                raw_emotion  = top_emotion(probs)
                # Majority vote: ambil emosi yang paling sering muncul
                # dalam 20 frame terakhir → tidak ikut-ikutan noise tiap frame
                emotion_votes.append(raw_emotion)
                if len(emotion_votes) > EMOTION_VOTE_N:
                    emotion_votes.pop(0)
                emotion_cls = Counter(emotion_votes).most_common(1)[0][0]

            # (e) Fuzzy setiap FUZZY_INTERVAL detik
            if (now - last_fuzzy_t) >= FUZZY_INTERVAL:
                try:
                    fuzzy_score = calculate_fuzzy_engagement(
                        simulation, ear_s, yaw_s, emotion_cls)
                except Exception as e:
                    print(f"[WARN] Fuzzy error: {e}")
                last_fuzzy_t = now
                print(f"[Fuzzy] EAR={ear_s:.3f}  Pose={yaw_s:.1f}°  "
                      f"Emosi={emotion_cls}  Score={fuzzy_score:.2f}")

            # (f) Overlay
            draw_hud(frame, ear_raw_s, ear_s, ear_baseline, yaw_s, emotion_cls, fuzzy_score, fps)
            draw_emotion_probs(frame, probs)

        else:
            cv2.putText(frame, "Wajah tidak terdeteksi",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ORANGE, 2)
            draw_hud(frame, 0.0, 0.0, ear_baseline, 0.0, "Netral", fuzzy_score, fps)

        cv2.imshow("Smart Engagement Tracker — Fase 3", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"snapshot_{int(now)}.png"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Snapshot: {fname}")
        elif key == ord("r"):
            # Kalibrasi ulang tanpa restart program
            print("[INFO] Memulai kalibrasi ulang...")
            ear_baseline = calibrate_ear(cap, landmarker)
            ear_buf.clear()
            print(f"[INFO] Baseline baru: {ear_baseline:.4f}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[INFO] Pipeline selesai.")


if __name__ == "__main__":
    main()
