import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
import cv2
os.makedirs("learning_lab", exist_ok=True)
os.makedirs("training_data", exist_ok=True)
st.set_page_config(
    page_title="Bovine Intel Pro",
    layout="wide",
    page_icon="🐄"
)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()

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
user_location = "Other"

# =========================
# DETECTION
# =========================
def detect_animals(img):
    results = yolo_model(img, conf=0.25, iou=0.45)

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

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)

        color = tuple(np.random.randint(0,255,3).tolist())

        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)

        # ✅ Only confidence (clean UI)
        cv2.putText(
            img_np,
            f"{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return img_np
# =========================
# CLASSIFICATION (OPEN-SET)
# =========================
def classify(img, user_location):
    if model is None:
        return "Model Not Loaded", 0.0, np.zeros(len(CLASS_NAMES))
    
    img = img.resize((224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet.preprocess_input(arr)

    preds = model.predict(arr)[0]

    top_idx = np.argsort(preds)[::-1]
    top1, top2 = preds[top_idx[0]], preds[top_idx[1]]

    label = CLASS_NAMES[top_idx[0]]

    # 🔥 OPEN SET LOGIC (unchanged)
    if top1 < 0.80 or (top1 - top2) < 0.25:
        return "🧬 Hybrid / Unknown", top1, preds

    # ✅ NEW: GEO BOOST
    if label in BREED_DATA:
        origin = BREED_DATA[label]["Origin"].lower()
        if user_location.lower() in origin:
            top1 = min(top1 + 0.10, 0.99)
    return label, top1, preds

# =========================
# UI
# =========================
# =========================
# NAVIGATION
# =========================
with st.sidebar:
    st.title("🐄 Bovine Intel")

    page = st.radio("Navigate", ["Dashboard", "Breed Analyzer", "Learning Lab"])

    st.markdown("---")

    # ✅ Geospatial context
    user_location = st.selectbox(
        "📍 Field Location",
        ["Andhra Pradesh", "Gujarat", "Punjab", "Haryana", "Maharashtra", "Rajasthan", "Other"]
        )
                
# =========================
# DASHBOARD
# =========================
if page == "Dashboard":
    st.title("📊 Dashboard")
    st.write("AI-powered cattle breed detection & classification")
    st.info("Use Breed Analyzer to detect animals and Learning Lab to improve model.")

# =========================
# BREED ANALYZER
# =========================
elif page == "Breed Analyzer":

    st.title("🔍 Breed Analyzer")
    
    input_type = st.radio(
    "Select Input Type",
    ["Upload Image", "Camera"],
    horizontal=True
    )

    img = None

    if input_type == "Upload Image":
        file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
        if file:
            img = Image.open(file).convert("RGB")
    
    elif input_type == "Camera":
        cam = st.camera_input("Capture Image")
        if cam:
            img = Image.open(cam).convert("RGB")
        
    if img is not None:
        
        try:
            # ensure valid PIL image
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
    
            st.image(img, use_container_width=True)
    
        except Exception:
            st.error("⚠ Invalid image. Please re-upload or recapture.")
            st.stop()
            st.image(img, use_container_width=True)

        if st.button("🚀 Analyze", use_container_width=True):
          
            boxes, scores = detect_animals(img)

            if len(boxes) == 0:
                st.warning("No cows detected")
                st.stop()

            # 🔹 Boxed Image
            boxed = draw_boxes(img, boxes, scores)
            st.image(boxed, use_container_width=True)
            st.success(f"{len(boxes)} cows detected")
            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()

            # 🔹 GRID VIEW
            cols = st.columns(3)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                crop = img.crop((x1, y1, x2, y2))

                label, conf, preds = classify(crop, user_location)

                # 🔥 AUTO SAVE UNKNOWN → LEARNING LAB
                if "Unknown" in label:
                    filename = f"learning_lab/unknown_{i}_{np.random.randint(10000)}.jpg"
                    if not os.path.exists(filename):
                        crop.save(filename)

                with cols[i % 3]:
                    st.image(crop, use_container_width=True)
                    st.markdown(f"**Animal {i+1}**")

                    if "Unknown" in label:
                        st.warning("🧬 Hybrid / Unknown")
                    else:
                        st.success(label)

                    st.caption(f"Confidence: {conf*100:.1f}%")

            st.divider()

            # 🔹 CHART
            st.subheader("Prediction Distribution")
            if len(boxes) > 0:
                first_crop = img.crop(tuple(map(int, boxes[0])))
                _, _, first_preds = classify(first_crop, user_location)
            
                st.bar_chart({
                    CLASS_NAMES[i]: float(first_preds[i])
                    for i in range(len(CLASS_NAMES))
                })

# =========================
# LEARNING LAB
# =========================
elif page == "Learning Lab":

    st.title("🧪 Learning Lab")

    images = [f for f in os.listdir("learning_lab") if f.endswith(".jpg")]

    if not images:
        st.info("No unknown samples yet")
        st.stop()

    selected = st.selectbox("Select Image", images)

    img_path = os.path.join("learning_lab", selected)
    img = Image.open(img_path)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Review Image", use_container_width=True)

    with col2:
        st.subheader("Annotate")

        label = st.selectbox("Select Correct Breed", ["Unknown"] + CLASS_NAMES)

        colA, colB = st.columns(2)
        
        # 🔵 Annotate Button
        with colA:
            if st.button("✅ Save Annotation"):
                save_dir = f"training_data/{label}"
                os.makedirs(save_dir, exist_ok=True)
    
                img.save(os.path.join(save_dir, selected))
                os.remove(img_path)
    
                st.success(f"Saved as {label}")
                st.rerun()

        # 🔴 Delete Button
        with colB:
            if st.button("🗑 Delete"):
                os.remove(img_path)
                st.warning("Deleted")
                st.rerun()

    st.divider()

    # 📸 Camera Input in Learning Lab
    st.subheader("Add New Sample")

    cam = st.camera_input("Capture new animal")

    if cam:
        new_img = Image.open(cam)
        filename = f"learning_lab/manual_{np.random.randint(10000)}.jpg"
        new_img.save(filename)
        st.success("Added to Learning Lab")
        st.rerun()
        
st.download_button("Download Report", data="Coming soon")
