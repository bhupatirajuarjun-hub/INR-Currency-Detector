import streamlit as st
import subprocess
import re
from PIL import Image
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "models/IndianCurrencyDetector/resnet18.onnx"
LABELS_PATH = "models/IndianCurrencyDetector/labels.txt"
INPUT_BLOB = "input_0"
OUTPUT_BLOB = "output_0"

st.set_page_config(page_title="Indian Currency Detector", layout="wide")

st.title("💵 Indian Currency Authentication System")
st.write("Upload an image to detect whether currency is REAL or FAKE using Jetson AI model.")

# ==========================
# PDF REPORT
# ==========================
def create_pdf(report_text):
    file_path = "currency_report.pdf"
    c = canvas.Canvas(file_path)
    text = c.beginText(40, 750)

    for line in report_text.split("\n"):
        text.textLine(line)

    c.drawText(text)
    c.save()

    return file_path

# ==========================
# RUN JETSON INFERENCE
# ==========================
def run_inference(image_path):

    output_image = "output.jpg"

    cmd = [
        "imagenet.py",
        f"--model={MODEL_PATH}",
        f"--labels={LABELS_PATH}",
        f"--input_blob={INPUT_BLOB}",
        f"--output_blob={OUTPUT_BLOB}",
        image_path,
        output_image
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    output = result.stdout + result.stderr

    # ==========================
    # PARSE RESULT
    # ==========================
    match = re.search(r"imagenet:\s+([\d.]+)%\s+class #\d+\s+\((.*?)\)", output)

    if match:
        confidence = float(match.group(1))
        label = match.group(2)
    else:
        label = "Unknown"
        confidence = 0.0

    return label, confidence, output_image, output

# ==========================
# UPLOAD IMAGE
# ==========================
uploaded_file = st.file_uploader("Upload Currency Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Save uploaded image
    input_path = "input.jpg"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(input_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:

        st.subheader("Prediction")

        with st.spinner("Running Jetson AI model..."):
            label, confidence, output_img, raw_output = run_inference(input_path)

        if label.lower() == "fake":
            st.error(f"Prediction: FAKE")
        else:
            st.success(f"Prediction: REAL")

        st.metric("Confidence", f"{confidence:.2f}%")

    # ==========================
    # REPORT
    # ==========================
    report = f"""
INDIAN CURRENCY AUTHENTICATION REPORT
=====================================

Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Prediction: {label}
Confidence: {confidence:.2f}%

Conclusion:
"""

    if label.lower() == "fake":
        report += "The currency appears to be COUNTERFEIT."
    else:
        report += "The currency appears to be AUTHENTIC."

    st.subheader("Analysis Report")
    st.text(report)

    pdf = create_pdf(report)

    with open(pdf, "rb") as f:
        st.download_button(
            "Download PDF Report",
            data=f,
            file_name="currency_report.pdf",
            mime="application/pdf"
        )

    st.success("Analysis complete.")
