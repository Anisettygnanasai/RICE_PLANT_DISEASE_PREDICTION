import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import pandas as pd

# Add root directory to path to import src modules
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.inference import PlantDiseasePredictor

# Page config
st.set_page_config(
    page_title="Plant Disease Predictor",
    page_icon="🌿",
    layout="centered"
)

# Initialize predictor (using caching so it loads once)
@st.cache_resource
def load_predictor():
    # Use best_model.pth if it exists, otherwise it will throw an error telling user to train.
    checkpoint_path = root_dir / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        return None
    return PlantDiseasePredictor(checkpoint_path=str(checkpoint_path))

predictor = load_predictor()

st.title("🌿 Plant Disease Predictor")
st.write("Upload an image of a plant leaf to detect potential diseases.")

if predictor is None:
    st.error("⚠️ Model checkpoint not found! Please train the model first by running `python -m src.train`.")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing image..."):
        try:
            # Predict
            result = predictor.predict(image)
            
            st.success("Analysis Complete!")
            
            # Results display
            st.subheader("Prediction Results")
            st.write(f"**Predicted Disease:** {result['predicted_class']}")
            st.write(f"**Confidence:** {result['confidence']:.2%}")
            
            # Rich Enterprise Details
            details = result.get('details', {})
            if details:
                st.info(f"📖 **Description:** {details.get('description', 'N/A')}")
                st.warning(f"⚠️ **Causes:** {details.get('causes', 'N/A')}")
                st.success(f"💡 **Prevention & Treatment:** {details.get('prevention', 'N/A')}")
            
            # Top-3 Chart
            st.subheader("Top Predictions")
            top3 = result['top3']
            df = pd.DataFrame(top3)
            # Create a simple bar chart
            st.bar_chart(df.set_index('class'))
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("Developed with PyTorch & Albumentations. Disclaimer: This tool is for demonstration purposes and general guidance only.")
