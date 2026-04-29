import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
import cv2
st.set_page_config(page_title="Bovine Intel Pro", layout="wide")
# =========================
# LOAD MODELS
# =========================
yolo_model = YOLO("yolov8s.pt")

MODEL_PATH = "breed_classifier_mobilenet (2).h5"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    return None

model = load_model()

# =========================
# DATA
# =========================
BREED_DATA = {
    "Gir": {"Type": "Cattle", "Origin": "Gujarat"},
    "Sahiwal": {"Type": "Cattle", "Origin": "Punjab"},
    "Kankrej": {"Type": "Cattle", "Origin": "Gujarat"},
    "Ongole": {"Type": "Cattle", "Origin": "Andhra Pradesh"},
    "Red_Sindhi": {"Type": "Cattle", "Origin": "Sindh"},
    "Murrah": {"Type": "Buffalo", "Origin": "Haryana"},
    "Nagpuri": {"Type": "Buffalo", "Origin": "Maharashtra"},
    "Jaffarabadi": {"Type": "Buffalo", "Origin": "Gujarat"},
    "Bhadawari": {"Type": "Buffalo", "Origin": "UP"},
    "Toda": {"Type": "Buffalo", "Origin": "Nilgiris"}
}
CLASS_NAMES = sorted(BREED_DATA.keys())

# =========================
# DETECTION
# =========================
def detect_animals(img):
    results = yolo_model(img, conf=0.3, iou=0.5)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    animals = []
    for box, cls, score in zip(boxes, classes, scores):
        if int(cls) == 19:  # cow class
            animals.append((box, score))

    # sort by confidence
    animals = sorted(animals, key=lambda x: x[1], reverse=True)

    final_boxes, final_scores = [], []

    for box, score in animals:
        x1, y1, x2, y2 = box
        keep = True

        for fb in final_boxes:
            fx1, fy1, fx2, fy2 = fb

            inter_x1 = max(x1, fx1)
            inter_y1 = max(y1, fy1)
            inter_x2 = min(x2, fx2)
            inter_y2 = min(y2, fy2)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (fx2 - fx1) * (fy2 - fy1)

            union = area1 + area2 - inter_area
            iou = inter_area / union if union > 0 else 0

            if iou > 0.5:
                keep = False
                break

        if keep:
            final_boxes.append(box)
            final_scores.append(score)

    return final_boxes, final_scores

# =========================
# DRAW BOXES
# =========================
def draw_boxes(img, boxes, scores):
    img_np = np.array(img)

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(int, box)
        color = tuple(np.random.randint(0,255,3).tolist())

        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_np, f"Cow {i+1} ({score:.2f})",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img_np

# =========================
# CLASSIFICATION (OPEN-SET)
# =========================
def classify(img):
    img = img.resize((224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet.preprocess_input(arr)

    preds = model.predict(arr)[0]

    top_idx = np.argsort(preds)[::-1]
    top1, top2 = preds[top_idx[0]], preds[top_idx[1]]

    label = CLASS_NAMES[top_idx[0]]

    # 🔥 OPEN SET LOGIC
    if top1 < 0.75 or (top1 - top2) < 0.2:
        return "🧬 Hybrid / Unknown", top1, preds

    return label, top1, preds

# =========================
# UI
# =========================

img_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if img_file:
    img = Image.open(img_file).convert("RGB")

    if st.button("Analyze"):

        boxes, scores = detect_animals(img)

        if len(boxes) == 0:
            st.warning("No cows detected")
            st.stop()

        # 🔹 show boxed image
        boxed = draw_boxes(img, boxes, scores)
        st.image(boxed, use_container_width=True)
        st.success(f"{len(boxes)} cows detected")

        st.divider()

        # 🔹 GRID VIEW
        cols = st.columns(3)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = img.crop((x1, y1, x2, y2))

            with cols[i % 3]:
                st.image(crop, use_container_width=True)

                label, conf, preds = classify(crop)

                if "Unknown" in label:
                    st.warning(label)
                else:
                    st.success(label)

                st.caption(f"Confidence: {conf*100:.1f}%")

        st.divider()

        # 🔹 PROBABILITY CHART (only once)
        st.subheader("Prediction Distribution")
        st.bar_chart({CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))})
