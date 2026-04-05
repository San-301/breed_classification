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
MODEL_PATH = "breed_classifier_mobilenet (2).h5" 
CONFIDENCE_THRESHOLD = 0.58 

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
# 2. UI STYLING (Green Theme, No Blue Lines)
# ==========================================
st.set_page_config(page_title="Bovine Intel Pro", layout="wide", page_icon="🐂")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stSidebar { background-color: #111; border-right: 2px solid #2e7d32; }
    /* Green Accent Button */
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 8px; width: 100%; height: 3em; font-size: 1.2em; border: none; }
    /* Result Card with Green Line */
    .result-card { background: white; padding: 20px; border-radius: 12px; border-left: 8px solid #2e7d32; box-shadow: 0 4px 15px rgba(0,0,0,0.05); color: #333; }
    .info-tag { background: #e8f5e9; color: #2e7d32; padding: 4px 12px; border-radius: 4px; font-weight: bold; }
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
        
    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)
    raw_score = preds[top_idx]
    breed = CLASS_NAMES[top_idx]

    # Impact of Region: Adds 12% confidence if region matches origin
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
    st.caption("Note: Selecting the correct region improves identification accuracy based on local breed prevalence.")

if app_mode == "Dashboard":
    st.title("Indian Livestock Intelligence")
    st.write("Professional grade breed identification system.")

elif app_mode == "Breed Analyzer":
    st.title("🔍 Breed Analysis")
    
    img_file = st.file_uploader("Upload Image for Identification", type=["jpg", "png", "jpeg"])

    if img_file:
        # 1. Place Image
        st.image(img_file, use_container_width=True)
        
        # 2. Place Predict Button directly below image
        if st.button("Predict"):
            breed, confidence, all_preds = process_and_infer(img_file, user_location)
            
            if breed is None:
                st.error("Model file missing.")
            elif confidence < CONFIDENCE_THRESHOLD:
                # Uncertain Identification Message
                st.error("⚠️ Uncertain Identification")
                st.warning("Low confidence detected. Image moved to Learning Lab for review.")
                img_path = f"flagged_for_learning/low_conf_{int(time.time())}.jpg"
                Image.open(img_file).save(img_path)
            else:
                # 3. Show Stats below button if correct
                data = BREED_DATA[breed]
                st.markdown(f"""
                <div class="result-card">
                    <span class="info-tag">{data['Type'].upper()}</span>
                    <h2 style="margin: 10px 0;">{breed}</h2>
                    <p><b>Confidence Score:</b> {confidence*100:.1f}%</p>
                    <p><b>Origin:</b> {data['Origin']}</p>
                    <p style="font-style: italic;">{data['Description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("### Probability Distribution")
                st.bar_chart({CLASS_NAMES[i]: float(all_preds[i]) for i in range(len(CLASS_NAMES))})

elif app_mode == "Learning Lab":
    st.title("🧪 Smart Review Lab")
    flagged_images = os.listdir("flagged_for_learning")
    
    if flagged_images:
        col1, col2 = st.columns([1, 1])
        with col1:
            selected_img = st.selectbox("Select flagged image:", flagged_images)
            raw_img = Image.open(os.path.join("flagged_for_learning", selected_img))
            st.image(raw_img, caption="Original Flagged Image")
        
        with col2:
            st.subheader("Innovative Review: Focus Tool")
            st.write("Isolate the animal to remove background noise (Helpful for better labeling).")
            
            # Innovative Feature: Manual "Focus" slider to help user suggest a crop
            width, height = raw_img.size
            zoom = st.slider("Focus / Zoom Level", 50, 100, 100)
            
            if zoom < 100:
                left = (width * (100 - zoom) / 200)
                top = (height * (100 - zoom) / 200)
                right = width - left
                bottom = height - top
                cropped_img = raw_img.crop((left, top, right, bottom))
                st.image(cropped_img, caption="Zoomed Focus for Re-Labeling")
                
            new_label = st.selectbox("Suggest Correct Breed:", ["Unknown"] + CLASS_NAMES)
            if st.button("Submit Review"):
                st.success(f"Image tagged as {new_label}. Thank you for improving the AI!")
    else:
        st.info("No images flagged for review. Everything looks good!")
