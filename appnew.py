import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

st.set_page_config(page_title="Counterfeit Currency Detector", layout="wide")

st.title("💵 Counterfeit Currency Detector")
st.write("Upload an Indian currency note for analysis.")

MODEL_PATH = "model/resnet18.onnx"
LABELS_PATH = "model/labels.txt"

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

# Preprocess image
    img = image.resize((224, 224))

    img = np.array(img)

    img = img.astype(np.float32)

# No normalization
# No division by 255

    img = np.transpose(img, (2, 0, 1))

    img = np.expand_dims(img, axis=0)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    output = session.run([output_name], {input_name: img})[0]

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    output = session.run([output_name], {input_name: img})[0]

    # Softmax
    exp = np.exp(output - np.max(output))
    probs = exp / np.sum(exp)

    prediction = np.argmax(probs)

    confidence = float(probs[0][prediction] * 100)

    with col2:

        st.subheader("Prediction")

        if labels[prediction].lower() == "real":
            st.success("✅ REAL NOTE")
        else:
            st.error("❌ FAKE NOTE")

        st.metric("Confidence", f"{confidence:.2f}%")

        st.write("Raw output:", output.tolist())
        st.write("Probabilities:", probs.tolist())

        st.write("Prediction index:", prediction)
        st.write("Labels:", labels)
