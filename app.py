import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Histopathologic Cancer Detection",
    page_icon="🔬",
    layout="wide",
)

# --- MODEL LOADING ---
# Use caching to load the model only once
@st.cache_resource
def load_histopathology_model():
    """Loads the pre-trained cancer detection model."""
    try:
        model = load_model('histopathology_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the 'histopathology_model.h5' file is in the same directory as app.py.")
        st.info("You can train a model using a separate training script or download a pre-trained one.")
        return None

model = load_histopathology_model()

# --- HELPER FUNCTIONS ---
def preprocess_image(image):
    """Preprocesses the uploaded image to be model-ready."""
    img = image.resize((96, 96))  # Model expects 96x96 images
    img_array = np.array(img)
    img_array = img_array / 255.0   # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

def generate_detailed_report(prediction_probability):
    """Generates a detailed report based on the prediction probability."""
    is_cancerous = prediction_probability > 0.5
    confidence = prediction_probability if is_cancerous else 1 - prediction_probability

    if is_cancerous:
        st.error("### Findings: Malignant (Cancerous) Tissue Detected")
        st.markdown(f"""
        **Confidence:** **{confidence:.2%}**
        
        **Interpretation:**
        The model has identified features within the tissue sample that are highly indicative of malignant cells. This classification is based on patterns learned from thousands of labeled histopathological images.
        
        **Key indicators may include:**
        - Irregular cell nuclei shape and size.
        - High density of cells (hypercellularity).
        - Presence of mitotic figures suggesting rapid cell division.
        
        **Next Steps:**
        This result warrants immediate review by a qualified pathologist for definitive diagnosis and to determine the cancer's type and grade.
        """)
    else:
        st.success("### Findings: Benign (Non-Cancerous) Tissue Detected")
        st.markdown(f"""
        **Confidence:** **{confidence:.2%}**

        **Interpretation:**
        The model's analysis suggests that the provided tissue sample does not contain malignant cells. The cellular structures appear normal and consistent with healthy tissue.

        **Key indicators observed:**
        - Uniform cell shapes and sizes.
        - Normal tissue architecture.
        - Absence of invasive patterns.

        **Next Steps:**
        While this result is positive, it should be confirmed as part of a comprehensive medical evaluation by a healthcare professional.
        """)
    
    st.warning("""
    **Disclaimer:** This is an AI-powered educational tool and **not a substitute for professional medical advice**. 
    The results are for informational purposes only and should not be used for self-diagnosis. 
    Always consult a qualified doctor for any health concerns.
    """)

# --- WEBSITE DESIGN ---
st.title("🔬 Histopathologic Cancer Detection AI")
st.markdown("Upload a histopathologic image (96x96 px) of lymph node tissue, and the AI will analyze it for signs of metastatic cancer.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("About This Project")
    st.markdown("""
    This web application uses a deep learning model (Convolutional Neural Network) to classify histopathologic images. 
    
    The model was trained on the **PatchCamelyon (PCam)** dataset to distinguish between images containing cancerous tissue and those that do not.
    """)
    st.header("How It Works")
    st.markdown("""
    1. **Upload an Image:** Use the file uploader below.
    2. **AI Analysis:** The image is pre-processed and fed into the trained model.
    3. **Get Report:** The model predicts the probability of cancer, and a detailed report is generated.
    """)

# --- MAIN CONTENT ---
uploaded_file = st.file_uploader("Choose a tissue sample image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Tissue Sample", use_column_width=True)
    
    with col2:
        st.info("Analyzing... Please wait.")
        try:
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)
            
            # Get prediction
            prediction = model.predict(processed_image)
            prediction_probability = prediction[0][0]
            
            # Generate and display the detailed report
            generate_detailed_report(prediction_probability)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

elif uploaded_file is not None and model is None:
    st.error("Model is not loaded. Cannot perform analysis.")