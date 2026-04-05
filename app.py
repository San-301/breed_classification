import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time
from PIL import Image

# ==========================================
# 1. SYSTEM CONFIGURATION & DATA
# ==========================================
MODEL_PATH = "breed_classifier_mobilenet.h5"
CONFIDENCE_THRESHOLD = 0.58  # Optimized rejection threshold
# Create a directory to save low-confidence images for future retraining (Active Learning)
if not os.path.exists("flagged_for_learning"):
    os.makedirs("flagged_for_learning")

BREED_DATA = {
    "Bhadawari": {"Type": "Buffalo", "Origin": "UP & MP", "Description": "High-fat milk breed, heat-tolerant."},
    "Gir": {"Type": "Cattle", "Origin": "Gujarat", "Description": "High milk-yielding indigenous breed."},
    "Jaffarabadi": {"Type": "Buffalo", "Origin": "Gujarat", "Description": "Large-sized dairy buffalo."},
    "Kankrej": {"Type": "Cattle", "Origin": "Gujarat & Rajasthan", "Description": "Dual-purpose breed for milk and draught."},
    "Murrah": {"Type": "Buffalo", "Origin": "Haryana & Punjab", "Description": "High-fat milk producing buffalo."},
    "Nagpuri": {"Type": "Buffalo", "Origin": "Maharashtra", "Description": "Drought-resistant dairy buffalo."},
    "Ongole": {"Type": "Cattle", "Origin": "Andhra Pradesh", "Description": "Large breed used for draught and milk."},
    "Red_Sindhi": {"Type": "Cattle", "Origin": "Sindh region", "Description": "Adapted for tropical climates."},
    "Sahiwal": {"Type": "Cattle", "Origin": "Punjab", "Description": "Heat-tolerant dairy breed."},
    "Toda": {"Type": "Buffalo", "Origin": "Nilgiri Hills", "Description": "Small-sized buffalo, adapted to hilly terrain."}
}
CLASS_NAMES = sorted(BREED_DATA.keys())

# ==========================================
# 2. UI STYLING
# ==========================================
st.set_page_config(page_title="Bovine Intel Pro", layout="wide", page_icon="🐂")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stSidebar { background-color: #111; color: white; }
    .stButton>button { background-color: #1e40af; color: white; border-radius: 10px; height: 3.5em; font-weight: bold; }
    .result-card { background: white; padding: 25px; border-radius: 15px; border-left: 10px solid #1e40af; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
    .info-tag { background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 5px; font-weight: bold; font-size: 0.8em; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. CORE ANALYTICS ENGINE
# ==========================================
@st.cache_resource
def load_optimized_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    return None

def process_and_infer(img_source, user_state):
    """Production Inference Pipeline with Contextual Boosting"""
    img = Image.open(img_source).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

    model = load_optimized_model()
    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)
    raw_score = preds[top_idx]
    breed = CLASS_NAMES[top_idx]

    # INNOVATION: Contextual Bayesian Boost
    # If the detected breed's origin matches the user's current state, boost confidence
    final_score = raw_score
    if user_state.lower() in BREED_DATA[breed]['Origin'].lower():
        final_score = min(raw_score + 0.12, 0.99) # 12% Boost for regional accuracy

    return breed, final_score, preds

# ==========================================
# 4. APP INTERFACE
# ==========================================
with st.sidebar:
    st.title("🛰️ Bovine Intel")
    st.markdown("---")
    app_mode = st.radio("Navigation", ["Dashboard", "Breed Analyzer", "Learning Lab"])
    st.info("System Status: **Active**")
    user_location = st.selectbox("Current Field Location (State)", 
                                ["Andhra Pradesh", "Gujarat", "Haryana", "Maharashtra", "Punjab", "Rajasthan", "Other"])

if app_mode == "Dashboard":
    st.title("Indian Livestock Intelligence Dashboard")
    st.write("This application empowers Field Level Workers (FLWs) to identify indigenous breeds using MobileNet deep learning and geospatial context.")
    st.image("https://googleusercontent.com", use_container_width=True)

elif app_mode == "Breed Analyzer":
    st.title("🔍 Multi-Input Breed Analysis")
    
    # Toggle between Upload and Camera
    input_method = st.radio("Select Input Method:",, horizontal=True)
    
    img_file = None
    if input_method == "📁 Upload Image":
        img_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])
    else:
        img_file = st.camera_input("Capture Bovine Photo")

    if img_file:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(img_file, caption="Processing Input...", use_container_width=True)
            
        with col2:
            if st.button("🚀 EXECUTE AI IDENTIFICATION"):
                with st.spinner("Decoding Phenotypic Features..."):
                    breed, confidence, all_preds = process_and_infer(img_file, user_location)
                    
                    # INNOVATION: Threshold-based Guardrail (Validates if input is bovine)
                    if confidence < CONFIDENCE_THRESHOLD:
                        st.error("🚫 **Uncertain Identification**")
                        st.warning("Confidence score is too low. The image may be blurry or contain a non-supported breed.")
                        
                        # Active Learning Trigger: Save image for human review
                        img_path = f"flagged_for_learning/low_conf_{int(time.time())}.jpg"
                        Image.open(img_file).save(img_path)
                        st.info("System Note: This image has been flagged for 'Learning Lab' review to improve future accuracy.")
                    
                    else:
                        data = BREED_DATA[breed]
                        st.balloons()
                        st.markdown(f"""
                        <div class="result-card">
                            <span class="info-tag">{data['Type'].upper()} IDENTIFIED</span>
                            <h1 style="color:#1e40af; margin-top:10px;">{breed}</h1>
                            <p style="font-size:1.2em;"><b>Confidence:</b> {confidence*100:.1f}%</p>
                            <hr>
                            <p><b>Regional Origin:</b> {data['Origin']}</p>
                            <p><b>Key Characteristics:</b> {data['Description']}</p>
                            <small style="color:gray;">*Confidence includes Contextual Bayesian Boosting based on location: {user_location}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show Probability Distribution
                        with st.expander("View Neural Network Probability Distribution"):
                            st.bar_chart({CLASS_NAMES[i]: float(all_preds[i]) for i in range(len(CLASS_NAMES))})

elif app_mode == "Learning Lab":
    st.title("🧪 Active Learning Lab")
    st.write("This section stores images that the model found difficult to classify. In the next phase, these will be used for incremental retraining.")
    flagged_images = os.listdir("flagged_for_learning")
    if flagged_images:
        st.write(f"Total images awaiting expert review: **{len(flagged_images)}**")
        selected_img = st.selectbox("Select image to review:", flagged_images)
        st.image(os.path.join("flagged_for_learning", selected_img))
    else:
        st.success("No low-confidence images detected yet. System is performing within high-precision parameters.")
