import streamlit as st
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from streamlit_lottie import st_lottie

# ---------------- Page Config ----------------
st.set_page_config(page_title="RoadGuard AI", layout="wide", page_icon="🛣️")

# ---------------- CSS Styling (Your Reference Style) ----------------
st.markdown("""
<style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1545143333-6382f1d5b2bc?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: rgba(0,0,0,0.5); 
        z-index: -1;
    }
    /* Translucent Cards */
    div[data-testid="stVerticalBlock"] > div:has(div.stImage) {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    h1, h2, h3, p, span, label {
        color: white !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .stProgress > div > div > div {
        background-color: #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Session State & Model ----------------
if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_ai_model():
    # Using MobileNetV2 (Lightweight for Web)
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_ai_model()

# ---------------- Utility Functions ----------------
def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

def predict_condition(image):
    # Image Preprocessing
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Run Inference
    predictions = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]
    return decoded[1].replace("_", " ").title(), decoded[2]

# Assets
scan_anim = load_lottie("https://assets5.lottiefiles.com/packages/lf20_t9gkkhz4.json")

# ---------------- Header ----------------
st.markdown("<h1 style='text-align:center;'>🛣️ RoadGuard AI Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Identify potholes, cracks, and road surface quality using Deep Learning.</p>", unsafe_allow_html=True)

# ---------------- Sidebar History ----------------
st.sidebar.markdown("### 📜 Scan History")
if st.sidebar.button("Clear Log"):
    st.session_state.history = []

for item in reversed(st.session_state.history):
    with st.sidebar.expander(f"{item['label']}"):
        st.image(item['img'], use_container_width=True)
        st.caption(f"Confidence: {item['conf']:.2%}")

# ---------------- Main Layout ----------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📸 Upload Surface")
    uploaded_file = st.file_uploader("Drop road image here...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Analyzed View", use_container_width=True)
        
        if st.button("🚀 Run AI Diagnosis", use_container_width=True, type="primary"):
            with st.spinner("Analyzing pixels..."):
                # Run Logic
                label, confidence = predict_condition(input_image)
                
                # Save to session
                st.session_state.current_result = {"label": label, "conf": confidence}
                st.session_state.history.append({"label": label, "conf": confidence, "img": input_image})
                st.rerun()

with col2:
    st.markdown("### 📊 Diagnostic Results")
    
    if "current_result" in st.session_state:
        res = st.session_state.current_result
        
        # Convert to standard python float for Streamlit compatibility
        conf_value = float(res['conf'])
        
        st.metric(label="Detected Condition", value=res['label'])
        st.write(f"**Confidence Level:** {conf_value:.2%}")
        
        # This is where the fix is applied
        st.progress(float(conf_value))
def predict_condition(image):
    # ... preprocessing code ...
predictions = model.predict(img_array)
decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]
    
# decoded[2] is the confidence score
return decoded[1].replace("_", " ").title(), float(decoded[2])
# ---------------- Footer ----------------
st.markdown("---")
st.caption("Engineered with TensorFlow & Streamlit • 2026 Edition")

