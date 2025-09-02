
import streamlit as st
from PIL import Image as PILImage, ImageDraw, ImageFont
import os
import numpy as np
import onnxruntime
from feedback_data_onnx import feedback_data  # Your feedback dictionary
import pandas as pd
from io import BytesIO

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Geotechnical Fault Detection", page_icon="🛠️")

st.title("🛠️ Geotechnical Fault Detection (ONNX + Maintenance Feedback)")
st.markdown("Upload an image to detect structural/geotechnical faults and receive actionable recommendations.")

ONNX_MODEL_PATH = 'best.onnx'

@st.cache_resource
def load_onnx_model(path):
    if os.path.exists(path):
        try:
            session = onnxruntime.InferenceSession(path, None)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            return session, input_name, output_name
        except Exception as e:
            st.error(f"Error loading ONNX model: {e}")
    else:
        st.error(f"ONNX model not found at {path}")
    return None, None, None

session, input_name, output_name = load_onnx_model(ONNX_MODEL_PATH)

CLASS_NAMES = [
    'Basket Corrosion Wall', 'Broken Basket', 'Bulging Face', 'Crack on Asphalt',
    'Deformation Wall', 'Expose foundation wall', 'Interface Opening', 'Long Crack GBA',
    'Long Crack Wall 1', 'Long Crack Wall 2', 'Mesh Crack Wall', 'Misalignment of wall',
    'Opening on GBA', 'Slope Deformation', 'Vegetation on Slope', 'Vegetation on Wall',
    'Vertical Crack GBA 1', 'Vertical Crack GBA 2', 'Vertical Crack Wall 1', 'Vertical Crack Wall 2'
]

# UI for image upload
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PILImage.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    if session:
        with st.spinner("🔍 Running inference and processing detections..."):
            img = np.array(image)
            img_resized = PILImage.fromarray(img).resize((640, 640))
            img_resized = np.array(img_resized)
            img_processed = img_resized[:, :, ::-1].transpose(2, 0, 1)
            img_processed = np.ascontiguousarray(img_processed).astype(np.float32) / 255.0
            img_processed = np.expand_dims(img_processed, 0)

            try:
                onnx_inputs = {input_name: img_processed}
                onnx_outputs = session.run([output_name], onnx_inputs)

                if not onnx_outputs or onnx_outputs[0] is None or len(onnx_outputs[0]) == 0:
                    st.warning("⚠️ ONNX model returned empty output. No predictions made.")
                    st.stop()

                try:
                    predictions = onnx_outputs[0].transpose(0, 2, 1)[0]
                except Exception as e:
                    st.error(f"❌ Failed to process ONNX output shape: {e}")
                    st.stop()

                confidence_threshold = 0.25
                boxes = predictions[:, :4]
                confidences = np.max(predictions[:, 4:], axis=1)
                class_ids = np.argmax(predictions[:, 4:], axis=1)

                valid_detections = confidences > confidence_threshold
                boxes = boxes[valid_detections]
                class_ids = class_ids[valid_detections]
                confidences = confidences[valid_detections]

                if len(boxes) == 0:
                    st.warning("⚠️ No valid detections above the confidence threshold.")
                    st.stop()

                original_width, original_height = image.size
                img_size_model = 640

                # Convert from center_x, center_y, width, height to x_min, y_min, x_max, y_max and scale to original image size
                boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * (original_width / img_size_model)  # x_min
                boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * (original_height / img_size_model) # y_min
                boxes[:, 2] = (boxes[:, 0] + boxes[:, 2] * (original_width / img_size_model))      # x_max
                boxes[:, 3] = (boxes[:, 1] + boxes[:, 3] * (original_height / img_size_model))     # y_max

                draw = ImageDraw.Draw(image)
                colors = {}
                results_data = [] # This will store ALL detections for CSV
                displayed_faults = {} # To track which fault types have been displayed

                # Collect all detections first
                all_detections = []
                for i in range(len(boxes)):
                    box = boxes[i]
                    class_id = int(class_ids[i])
                    confidence = confidences[i]
                    class_name = CLASS_NAMES[class_id]
                    
                    # Get risk weight from feedback data (default to 5 if not found)
                    fault_key = class_name.lower().strip()
                    risk_weight = feedback_data.get(fault_key, {}).get("risk_weight", 5)
                    
                    # Calculate SWRI (Severity-Weighted Risk Index)
                    swri_score = confidence * risk_weight

                    all_detections.append({
                        "box": box,
                        "class_id": class_id,
                        "confidence": confidence,
                        "class_name": class_name,
                        "risk_weight": risk_weight,
                        "swri": swri_score
                    })
                
                # Sort detections by SWRI score for prioritization (highest risk first)
                all_detections.sort(key=lambda x: x['swri'], reverse=True)

                # Process detections for display and CSV
                for det in all_detections:
                    box = det["box"]
                    class_id = det["class_id"]
                    confidence = det["confidence"]
                    class_name = det["class_name"]
                    risk_weight = det["risk_weight"]
                    swri_score = det["swri"]
                    
                    # Format label to include SWRI
                    label = f"{class_name}: {confidence:.2f} (Risk: {swri_score:.2f})"

                    # Add to results_data for CSV (all detections)
                    fault_key = class_name.lower().strip()
                    feedback = feedback_data.get(fault_key)

                    x_min = int(box[0])
                    y_min = int(box[1])
                    x_max = int(box[2])
                    y_max = int(box[3])

                    results_data.append({
                        "Image Filename": uploaded_file.name,
                        "Fault": class_name,
                        "Confidence": round(float(confidence), 2),
                        "Risk Weight": risk_weight,
                        "SWRI Score": round(float(swri_score), 2),
                        "Score": feedback['score'] if feedback else "N/A",
                        "Severity": feedback['severity'] if feedback else "N/A",
                        "Recommendation": feedback['recommendation'] if feedback else "N/A",
                        "Priority": feedback['priority'] if feedback else "N/A",
                        "X_min": x_min,
                        "Y_min": y_min,
                        "X_max": x_max,
                        "Y_max": y_max
                    })

                    # Only display one instance per class on the image
                    if class_name not in displayed_faults:
                        displayed_faults[class_name] = True

                        if class_id not in colors:
                            colors[class_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                        color = colors[class_id]

                        # Draw bounding box
                        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3) # Increased width for visibility

                        # Draw text with background for better visibility
                        try:
                            font = ImageFont.truetype("arial.ttf", 20) # Adjust font size as needed
                        except IOError:
                            font = ImageFont.load_default()

                        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                        text_width = right - left
                        text_height = bottom - top
                        text_x = box[0]
                        text_y = box[1] - text_height - 5 if box[1] - text_height - 5 > 0 else box[1] + 5 # Position above or below box

                        draw.rectangle([(text_x, text_y), (text_x + text_width, text_y + text_height)], fill=color)
                        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font) # White text on colored background

                with col2:
                    st.image(image, caption="Detected Results (Prioritized by SWRI)", use_column_width=True)

                st.markdown("--- ")
                st.subheader("Detection Details and Recommendations")
                
                # Display SWRI explanation
                with st.expander("ℹ️ About SWRI (Severity-Weighted Risk Index)"):
                    st.markdown("""
                    The **SWRI** prioritizes defects based on both detection confidence and engineering risk:
                    - **SWRI = Confidence × Risk Weight**
                    - Higher values indicate more critical defects needing immediate attention
                    - This ensures high-risk defects are addressed first, even with lower confidence
                    """)

                # Display prioritized faults by SWRI
                st.subheader("🔴 Risk-Based Prioritization (SWRI)")
                
                if all_detections:
                    # Create a DataFrame for the prioritized table
                    priority_df = pd.DataFrame([
                        {
                            "Fault": det["class_name"],
                            "Confidence": f"{det['confidence']:.2f}",
                            "Risk Weight": det["risk_weight"],
                            "SWRI Score": f"{det['swri']:.2f}",
                            "Priority": feedback_data.get(det["class_name"].lower().strip(), {}).get("priority", "N/A")
                        }
                        for det in all_detections
                        if det["class_name"] not in displayed_faults or displayed_faults[det["class_name"]]  # Show only one per class
                    ])
                    
                    # Remove duplicates while keeping the first occurrence (highest SWRI)
                    priority_df = priority_df.drop_duplicates(subset="Fault", keep="first")
                    
                    st.dataframe(priority_df, use_container_width=True)
                    
                    # Display feedback for all unique detected fault types (one entry per type)
                    st.subheader("Detailed Recommendations")
                    unique_faults_displayed = set()
                    for det in all_detections:
                        class_name = det["class_name"]
                        if class_name not in unique_faults_displayed:
                            unique_faults_displayed.add(class_name)
                            fault_key = class_name.lower().strip()
                            feedback = feedback_data.get(fault_key)

                            if feedback:
                                st.markdown(f"### 🧱 Fault: `{class_name}`")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.info(f"**Confidence**: `{det['confidence']:.2f}`")
                                    st.info(f"**Risk Weight**: `{det['risk_weight']}`")
                                    st.info(f"**SWRI Score**: `{det['swri']:.2f}`")
                                with col_b:
                                    st.info(f"**Score**: `{feedback['score']}`")
                                    st.info(f"**Severity**: `{feedback['severity']}`")
                                    st.warning(f"**Priority**: `{feedback['priority']}`")
                                st.success(f"**Recommendation**: {feedback['recommendation']}")
                                st.markdown("---")
                            else:
                                st.error(f"⚠️ No feedback found for `{class_name}`")
                                st.markdown("---")
                else:
                    st.info("No defects detected above the confidence threshold.")

                # Download buttons
                st.subheader("Download Results")
                col_dl1, col_dl2 = st.columns(2)

                with col_dl1:
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=byte_im,
                        file_name="detected_faults.png",
                        mime="image/png"
                    )

                with col_dl2:
                    if results_data:
                        df = pd.DataFrame(results_data)
                        csv = df.to_csv(index=False).encode('utf-8')

                        st.download_button(
                            label="📄 Download All Detections + Feedback as CSV",
                            data=csv,
                            file_name="fault_feedback_results.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"❌ Error during ONNX inference or processing: {e}")
                st.exception(e) # Display full exception traceback

    else:
        st.warning("⚠️ Model session is not active. Please ensure 'best.onnx' is in the correct directory.")

st.markdown("--- ")
st.markdown("🔗 Powered by YOLOv8 ONNX + Streamlit + Maintenance Intelligence + SWRI")

# Custom CSS for better UI/UX
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        border: 1px solid #007bff;
        color: #007bff;
        background-color: #ffffff;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #007bff;
        color: #ffffff;
    }
    .stFileUploader label {
        font-size: 1.1rem;
        font-weight: bold;
        color: #333;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    h1 {
        color: #007bff;
        text-align: center;
    }
    h3 {
        color: #333;
    }
    .css-1d3f8as {
        padding-top: 0rem;
    }
    /* Highlight high-risk items */
    .high-risk {
        background-color: #ffcccc !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)