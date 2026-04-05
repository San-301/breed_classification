import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time
import shutil
from PIL import Image

# ==========================================
# 1. SYSTEM CONFIGURATION & DATA
# ==========================================
MODEL_PATH = "breed_classifier_mobilenet (2).h5" 
CONFIDENCE_THRESHOLD = 0.58 

# Ensure directories exist for GitHub/Streamlit Cloud environment
for folder in ["flagged_for_learning", "training_queue"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

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
# 2. UI STYLING (Green Theme, Colored Buttons)
# ==========================================
# ==========================================
# 2. UI STYLING (Forced Color Fix)
# ==========================================
st.set_page_config(page_title="Bovine Intel Pro", layout="wide", page_icon="🐂")

# This new CSS targets buttons specifically by their container and forced states
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stSidebar { background-color: #111; border-right: 2px solid #2e7d32; }
    
    /* 1. Global Button Style (Green) */
    .stButton>button { 
        background-color: #2e7d32 !important; 
        color: white !important; 
        border-radius: 8px !important; 
        border: none !important;
    }

    /* 2. Target Review Button (Blue) using its unique key */
    div[data-testid="stSidebarNav"] + div div[data-testid="column"]:nth-of-type(1) button {
        background-color: #1e40af !important;
    }

    /* 3. Target Delete Button (Red) using its unique key */
    div[data-testid="stSidebarNav"] + div div[data-testid="column"]:nth-of-type(2) button {
        background-color: #dc3545 !important;
    }

    /* Hover effects to make it feel professional */
    .stButton>button:hover { opacity: 0.8; border: 1px solid white !important; }
    
    .result-card { background: white; padding: 20px; border-radius: 12px; border-left: 8px solid #2e7d32; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_optimized_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    return None

def process_and_infer(img_source, user_state):
    img = Image.open(img_source).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

    model = load_optimized_model()
    if model is None: return None, 0, None
        
    # FIX: Use [0] to extract the 1D prediction array and avoid IndexError
    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)
    raw_score = preds[top_idx]
    breed = CLASS_NAMES[top_idx]

    final_score = raw_score
    if user_state.lower() in BREED_DATA[breed]['Origin'].lower():
        final_score = min(raw_score + 0.12, 0.99) 

    return breed, final_score, preds

# ==========================================
# 4. APP INTERFACE
# ==========================================
with st.sidebar:
    st.title("🛰️ Bovine Intel")
    app_mode = st.radio("Go To:", ["Dashboard", "Breed Analyzer", "Learning Lab"])
    st.markdown("---")
    user_location = st.selectbox("Current Field Location", 
                                ["Andhra Pradesh", "Gujarat", "Haryana", "Maharashtra", "Punjab", "Rajasthan", "Other"])

if app_mode == "Dashboard":
    st.title("Indian Livestock Intelligence")
    st.write("Deep learning identification for indigenous breeds.")

elif app_mode == "Breed Analyzer":
    st.title("🔍 Breed Analysis")
    
    input_type = st.radio("Input Source:", ["Upload File", "Camera"], horizontal=True)
    
    if input_type == "Upload File":
        img_file = st.file_uploader("Choose a photo", type=["jpg", "png", "jpeg"])
    else:
        img_file = st.camera_input("Capture live photo")

    if img_file:
        st.image(img_file, use_container_width=True)
        if st.button("Predict"):
            breed, confidence, all_preds = process_and_infer(img_file, user_location)
            
            if breed is None:
                st.error("Model file not found.")
            elif confidence < CONFIDENCE_THRESHOLD:
                st.error("⚠️ Uncertain Identification")
                st.warning("Low confidence. Image moved to Learning Lab for review.")
                img_path = f"flagged_for_learning/low_conf_{int(time.time())}.jpg"
                Image.open(img_file).save(img_path)
            else:
                data = BREED_DATA[breed]
                st.markdown(f"""
                <div class="result-card">
                    <span class="info-tag">{data['Type'].upper()}</span>
                    <h2 style="margin: 10px 0;">{breed}</h2>
                    <p><b>Confidence:</b> {confidence*100:.1f}%</p>
                    <p><b>Origin:</b> {data['Origin']}</p>
                    <p><i>{data['Description']}</i></p>
                </div>
                """, unsafe_allow_html=True)
                st.write("### Probability Distribution")
                st.bar_chart({CLASS_NAMES[i]: float(all_preds[i]) for i in range(len(CLASS_NAMES))})

elif app_mode == "Learning Lab":
    st.title("🧪 Smart Review Lab")
    flagged_images = [f for f in os.listdir("flagged_for_learning") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if flagged_images:
        col_view, col_tool = st.columns(2)
        with col_view:
            selected_img = st.selectbox("Select image to review:", flagged_images)
            img_path = os.path.join("flagged_for_learning", selected_img)
            raw_img = Image.open(img_path)
            st.image(raw_img, caption="Original Flagged Image")
        
        with col_tool:
            st.subheader("Innovative Review: Focus Tool")
            width, height = raw_img.size
            zoom = st.slider("Focus / Zoom Level", 50, 100, 100)
            
            final_img = raw_img
            if zoom < 100:
                l = (width * (100 - zoom) / 200)
                t = (height * (100 - zoom) / 200)
                final_img = raw_img.crop((l, t, width-l, height-t))
                st.image(final_img, caption="Cropped Focus Area")
                
            new_label = st.selectbox("Suggest Correct Breed:", ["Unknown"] + CLASS_NAMES)
            
            btn_col_rev, btn_col_del = st.columns(2)
            with btn_col_rev:
                # BLUE BUTTON: Saves to training_queue/[BreedName]/
                if st.button("Submit Review"):
                    target_dir = os.path.join("training_queue", new_label)
                    os.makedirs(target_dir, exist_ok=True)
                    final_img.save(os.path.join(target_dir, f"rev_{selected_img}"))
                    os.remove(img_path)
                    st.success(f"Verified as {new_label}. AI data improved!")
                    time.sleep(1)
                    st.rerun()

            with btn_col_del:
                # RED BUTTON: Deletes permanently
                if st.button("🗑️ Delete"):
                    os.remove(img_path)
                    st.warning("Image permanently deleted.")
                    time.sleep(1)
                    st.rerun()
    else:
        st.info("No images flagged for review.")
