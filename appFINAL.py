import streamlit as st
import subprocess
import tempfile
import shutil
import re
from PIL import Image

st.set_page_config(page_title="Counterfeit Currency Detector", layout="centered")

st.title("💵 Counterfeit Currency Detector")
st.write("Upload an image of an Indian banknote to analyze whether it is genuine or counterfeit.")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):

        with st.spinner("Running AI analysis..."):

            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            image.save(temp.name)

            command = [
                "/usr/local/bin/imagenet.py",
                "--model=/home/nvidia/jetson-inference/python/training/classification/models/IndianCurrencyDetector/resnet18.onnx",
                "--labels=/home/nvidia/jetson-inference/python/training/classification/models/IndianCurrencyDetector/labels.txt",
                "--input_blob=input_0",
                "--output_blob=output_0",
                temp.name,
                "/tmp/output.jpg"
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            output = result.stdout + result.stderr

            match = re.search(
                r"imagenet:\s+([\d\.]+)%\s+class\s+#\d+\s+\((.*?)\)",
                output
            )

            if match:

                confidence = float(match.group(1))
                prediction = match.group(2)

                if prediction.lower() == "fake":
                    st.error(f"❌ COUNTERFEIT NOTE\n\nConfidence: {confidence:.2f}%")
                else:
                    st.success(f"✅ GENUINE NOTE\n\nConfidence: {confidence:.2f}%")

                st.progress(min(confidence / 100, 1.0))

                st.subheader("AI Analysis")

                st.write(
                    f"""
The neural network classified this banknote as **{prediction.upper()}**
with **{confidence:.2f}% confidence**.

This prediction is based on visual patterns learned during training.
The AI analyzes the overall appearance of the note rather than checking
individual security features directly.
"""
                )

                st.subheader("Recommended Manual Checks")

                st.markdown("""
- ✓ Watermark
- ✓ Security Thread
- ✓ See-through Register
- ✓ Micro Lettering
- ✓ Color-shifting Ink
- ✓ Serial Number Alignment
""")

            else:
                st.error("Could not analyze the image.")
                st.text(output)
