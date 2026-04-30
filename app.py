import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time
from PIL import Image
from ultralytics import YOLO

# ==============================
# LOAD MODELS
# ==============================
yolo_model = YOLO("yolov8s.pt")

MODEL_PATH = "breed_classifier_mobilenet (2).h5"

# Create folders
os.makedirs("flagged_for_learning", exist_ok=True)
os.makedirs("training_queue", exist_ok=True)

# Breed data
BREED_DATA = {
    "Bhadawari": {}, "Gir": {}, "Jaffarabadi": {},
    "Kankrej": {}, "Murrah": {}, "Nagpuri": {},
    "Ongole": {}, "Red_Sindhi": {}, "Sahiwal": {}, "Toda": {}
}
CLASS_NAMES = sorted(BREED_DATA.keys())

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    return None

# ==============================
# YOLO DETECTION
# ==============================
def detect_animals(img):
    results = yolo_model(img, conf=0.25)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    cow_boxes = []
    cow_scores = []

    for box, cls, score in zip(boxes, classes, scores):
        if int(cls) == 19:  # cow class
            cow_boxes.append(box)
            cow_scores.append(score)

    return cow_boxes, cow_scores

# ==============================
# DRAW BOXES
# ==============================
def draw_boxes(img, boxes, scores):
    img_np = np.array(img)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img_np, f"{score*100:.1f}%", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return img_np

# ==============================
# CLASSIFICATION
# ==============================
def classify(img, user_location):
    model = load_model()
    if model is None:
        return None, 0, None

    img = img.resize((224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet.preprocess_input(arr)

    preds = model.predict(arr)[0]

    top_idx = np.argsort(preds)[-3:][::-1]
    top1, top2 = preds[top_idx[0]], preds[top_idx[1]]

    # Decision logic
    if top1 < 0.6:
        if (top1 - top2) < 0.1:
            label = "Hybrid"
        else:
            label = "Unknown"
    elif (top1 - top2) < 0.15:
        label = "Ambiguous"
    else:
        label = CLASS_NAMES[top_idx[0]]

    confidence = float(top1)

    # Geo boost
    if label in BREED_DATA:
        if user_location.lower() in label.lower():
            confidence = min(confidence + 0.1, 0.99)

    return label, confidence, preds

# ==============================
# UI CONFIG
# ==============================
st.set_page_config(layout="wide")

with st.sidebar:
    app_mode = st.radio("Menu", ["Dashboard", "Analyzer", "Learning Lab"])
    user_location = st.selectbox("Location", ["Andhra Pradesh","Gujarat","Punjab","Other"])

# ==============================
# DASHBOARD
# ==============================
if app_mode == "Dashboard":
    st.title("🐄 Bovine Intelligence System")
    st.write("Detect and classify cattle breeds")

# ==============================
# ANALYZER
# ==============================
elif app_mode == "Analyzer":

    st.title("🔍 Breed Analyzer")

    input_type = st.radio("Input", ["Upload", "Camera"], horizontal=True)

    file = st.file_uploader("Upload Image", type=["jpg","png"]) \
        if input_type=="Upload" else st.camera_input("Capture")

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

        if st.button("Analyze"):

            with st.spinner("Analyzing..."):

                boxes, scores = detect_animals(img)

                if len(boxes) == 0:
                    st.error("🚫 No cows detected")
                else:

                    # Draw global image
                    boxed = draw_boxes(img, boxes, scores)
                    st.image(boxed, caption="Detected")

                    cols = st.columns(len(boxes))

                    for i,(box,col) in enumerate(zip(boxes, cols)):
                        x1,y1,x2,y2 = map(int, box)
                        crop = img.crop((x1,y1,x2,y2))

                        label, conf, preds = classify(crop, user_location)

                        with col:
                            st.image(crop)

                            st.write(f"Confidence: {conf*100:.1f}%")

                            # Flag wrong cases
                            if label in ["Unknown","Hybrid","Ambiguous"]:
                                path = f"flagged_for_learning/{time.time()}.jpg"
                                crop.save(path)

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
