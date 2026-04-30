import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import pandas as pd

# ==============================
# LOAD MODELS
# ==============================
yolo_model = YOLO("yolov8s.pt")
MODEL_PATH = "breed_classifier_mobilenet (2).h5"

# ==============================
# FOLDERS
# ==============================
os.makedirs("flagged_for_learning", exist_ok=True)
os.makedirs("training_queue", exist_ok=True)

# ==============================
# BREED DATA
# ==============================
BREED_DATA = {
    "Bhadawari": {}, "Gir": {}, "Jaffarabadi": {},
    "Kankrej": {}, "Murrah": {}, "Nagpuri": {},
    "Ongole": {}, "Red_Sindhi": {}, "Sahiwal": {}, "Toda": {}
}

BREED_ORIGIN = {
    "Murrah": ["haryana", "punjab"],
    "Gir": ["gujarat"],
    "Sahiwal": ["punjab"],
    "Ongole": ["andhra"],
    "Kankrej": ["gujarat", "rajasthan"],
    "Nagpuri": ["maharashtra"],
}

CLASS_NAMES = sorted(BREED_DATA.keys())
MAX_ANIMALS = 4

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    return None

model = load_model()

# ==============================
# IOU FUNCTION (NMS)
# ==============================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)

# ==============================
# DETECTION
# ==============================
def detect_animals(img):
    results = yolo_model(img, conf=0.35)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    img_w, img_h = img.size
    img_area = img_w * img_h

    center_x, center_y = img_w / 2, img_h / 2

    candidates = []

    for box, cls, score in zip(boxes, classes, scores):
        if int(cls) == 19:
            x1, y1, x2, y2 = box

            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / img_area

            if area_ratio < 0.05:
                continue

            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2

            dist = np.sqrt((box_center_x - center_x)**2 + (box_center_y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            center_score = 1 - (dist / max_dist)

            priority = (area_ratio * 0.5) + (score * 0.2) + (center_score * 0.3)

            candidates.append((box, score, priority))

    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

    # 🔥 NMS
    filtered_boxes = []
    filtered_scores = []

    for box, score, _ in candidates:
        keep = True
        for fbox in filtered_boxes:
            if iou(box, fbox) > 0.5:
                keep = False
                break
        if keep:
            filtered_boxes.append(box)
            filtered_scores.append(score)

    # limit
    filtered_boxes = filtered_boxes[:MAX_ANIMALS]
    filtered_scores = filtered_scores[:MAX_ANIMALS]

    return filtered_boxes, filtered_scores

# ==============================
# DRAW BOXES
# ==============================
def draw_boxes(img, boxes, scores):
    img_np = np.array(img)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{score*100:.1f}%"
        cv2.putText(img_np, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    img_pil = Image.fromarray(img_np)

    img_pil = ImageEnhance.Sharpness(img_pil).enhance(1.8)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(1.2)
    img_pil = ImageEnhance.Brightness(img_pil).enhance(1.05)

    return img_pil

# ==============================
# CLASSIFICATION
# ==============================
def classify(img, user_location):
    if model is None:
        return None, 0, None

    img = img.resize((224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet.preprocess_input(arr)

    preds = model.predict(arr)[0]

    top_idx = np.argsort(preds)[-3:][::-1]
    top1, top2 = preds[top_idx[0]], preds[top_idx[1]]

    if top1 < 0.65:
        if (top1 - top2) < 0.12:
            label = "Possible Hybrid Breed"
        else:
            label = "Unknown"
    elif (top1 - top2) < 0.18:
        label = "Possible Hybrid Breed"
    else:
        label = CLASS_NAMES[top_idx[0]]

    confidence = float(top1)

    # calibration
    confidence = 0.6 + (confidence * 0.4)
    confidence = min(confidence, 0.92)

    loc = user_location.lower()

    if label in BREED_ORIGIN:
        if any(region in loc for region in BREED_ORIGIN[label]):
            confidence += 0.05
        else:
            confidence -= 0.08
    else:
        confidence -= 0.03

    # special rule
    if label == "Gir" and "andhra" in loc:
        label = "Possible Hybrid Breed"
        confidence -= 0.1

    confidence = max(0.3, min(confidence, 0.92))

    return label, confidence, preds

# ==============================
# UI CONFIG
# ==============================
st.set_page_config(layout="wide")

st.markdown("""
<style>
div.stButton > button {border-radius:8px;font-weight:bold;}
div[data-testid="column"]:nth-of-type(1) button {background-color:#1e40af;color:white;}
div[data-testid="column"]:nth-of-type(2) button {background-color:#dc3545;color:white;}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    app_mode = st.radio("Menu", ["Dashboard", "Analyzer", "Learning Lab"])
    user_location = st.selectbox("Location",
        ["Andhra Pradesh","Gujarat","Punjab","Haryana","Rajasthan","Maharashtra","Other"])

# ==============================
# DASHBOARD
# ==============================
if app_mode == "Dashboard":
    st.title("🐄 Bovine Intelligence System")
    st.success("✔ Multi-animal detection ✔ Hybrid detection ✔ Reports")

# ==============================
# ANALYZER
# ==============================
elif app_mode == "Analyzer":

    st.title("🔍 Breed Analyzer")

    input_type = st.radio("Input", ["Upload", "Camera"], horizontal=True)

    file = st.file_uploader("Upload Image", type=None) \
        if input_type=="Upload" else st.camera_input("Capture")

    if file:
        try:
            img = Image.open(file).convert("RGB")
        except:
            st.error("Invalid image file")
            st.stop()

        st.image(img, use_container_width=True)

        if st.button("Analyze"):
            with st.spinner("Analyzing..."):

                boxes, scores = detect_animals(img)

                if len(boxes) == 0:
                    st.error("🚫 No animals detected")
                else:
                    st.markdown("### 🧠 Detection Output")
                    boxed = draw_boxes(img, boxes, scores)
                    st.image(boxed, use_container_width=True)

                    cols = st.columns(min(len(boxes), 4))
                    results_list = []

                    for idx, (box, col) in enumerate(zip(boxes, cols)):
                        x1, y1, x2, y2 = map(int, box)
                        crop = img.crop((x1, y1, x2, y2)).resize((250,250))

                        label, conf, preds = classify(crop, user_location)
                        results_list.append((idx+1, label, conf))

                        with col:
                            color = "green" if conf>0.75 else "orange" if conf>0.6 else "red"

                            st.markdown(f"""
                            <div style="border:2px solid {color};border-radius:10px;padding:10px;text-align:center;">
                            """, unsafe_allow_html=True)

                            st.image(crop)
                            st.markdown(f"### {label}")
                            st.markdown(f"Confidence: **{conf*100:.1f}%**")
                            st.markdown("</div>", unsafe_allow_html=True)

                            if label in ["Unknown","Possible Hybrid Breed"]:
                                crop.save(f"flagged_for_learning/{time.time()}.jpg")

                    # Report
                    df = pd.DataFrame([
                        {"Animal":r[0],"Prediction":r[1],"Confidence":f"{r[2]*100:.2f}%"}
                        for r in results_list
                    ])

                    st.download_button("📥 Download Report",
                        df.to_csv(index=False).encode(),
                        "report.csv")

# ==============================
# LEARNING LAB
# ==============================
elif app_mode == "Learning Lab":

    st.title("🧪 Learning Lab")

    images = os.listdir("flagged_for_learning")

    if not images:
        st.info("No flagged images")
    else:
        selected = st.selectbox("Select Image", images)
        path = os.path.join("flagged_for_learning", selected)
        img = Image.open(path)

        st.image(img)

        label = st.selectbox("Correct Label", ["Unknown"]+CLASS_NAMES)

        c1,c2 = st.columns(2)

        with c1:
            if st.button("Submit"):
                save_dir = f"training_queue/{label}"
                os.makedirs(save_dir, exist_ok=True)
                img.save(f"{save_dir}/{selected}")
                os.remove(path)
                st.success("Saved")
                st.rerun()

        with c2:
            if st.button("Delete"):
                os.remove(path)
                st.warning("Deleted")
                st.rerun()
