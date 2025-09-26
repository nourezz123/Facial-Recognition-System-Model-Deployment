import streamlit as st
import cv2
import numpy as np
import face_recognition
import pickle
from PIL import Image
import datetime
import pandas as pd
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="🕵️‍♂️ Face Verification System",
    page_icon="🛡️",
    layout="wide",
)

EMBEDDING_FILE = "face_embeddings.pkl"
LOG_FILE = "verification_log.csv"

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
@st.cache_resource
def load_embeddings():
    try:
        with open(EMBEDDING_FILE, "rb") as f:
            data = pickle.load(f)
        return np.array(data["embeddings"]), np.array(data["labels"])
    except Exception:
        st.error("⚠️ No valid embeddings file found. Please train/enroll faces first.")
        return np.array([]), np.array([])


# -----------------------------
# IMAGE PREPARATION
# -----------------------------
def prepare_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 2:  # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


# -----------------------------
# FACE RECOGNITION
# -----------------------------
def recognize_faces(image, known_embeddings, known_labels, tolerance=0.6, conf_threshold=75):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    results = []
    for encoding, (top, right, bottom, left) in zip(encodings, boxes):
        distances = face_recognition.face_distance(known_embeddings, encoding)
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        # Confidence scaling: 0 → 100% based on tolerance
        confidence = max(0, 100 * (1 - best_distance / tolerance))

        if best_distance < tolerance and confidence >= conf_threshold:
            name = known_labels[best_idx]
            results.append(("✅ Verified", name, confidence, (top, right, bottom, left)))
        else:
            results.append(("❌ Non Verified", None, confidence, (top, right, bottom, left)))

    return results


# -----------------------------
# SAVE HISTORY
# -----------------------------
def save_history(name, confidence):
    entry = {
        "Name": name if name else "Unknown",
        "Confidence (%)": f"{confidence:.2f}",
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state.history.append(entry)

    # Persist to CSV
    df = pd.DataFrame([entry])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)


# -----------------------------
# APP START
# -----------------------------
embeddings, labels = load_embeddings()
if embeddings.size == 0:
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image("logo.jpg", width=150)
st.sidebar.title("🔹 Face Verification")
st.sidebar.markdown("👋 Welcome to our Face Verification System")
st.sidebar.markdown("---")

confidence_slider = st.sidebar.slider("Confidence Threshold", min_value=50, max_value=100, value=75, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("📅 Session Info:")
st.sidebar.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# -----------------------------
# MAIN PAGE
# -----------------------------
st.title("🛡️ Face Verification System")

tab1, tab2 = st.tabs(["📂 Upload Image", "📷 Use Camera"])

with tab1:
    uploaded_file = st.file_uploader("📷 Upload a photo for verification", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = prepare_image(pil_image)

        results = recognize_faces(image_np, embeddings, labels, conf_threshold=confidence_slider)

        # Draw bounding boxes
        for status, name, confidence, (top, right, bottom, left) in results:
            color = (0, 255, 0) if status.startswith("✅") else (255, 0, 0)
            cv2.rectangle(image_np, (left, top), (right, bottom), color, 2)
            cv2.putText(image_np, f"{status} {confidence:.1f}%", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if status.startswith("✅"):
                st.success(f"{status}: {name} | Confidence: {confidence:.2f}%")
                save_history(name, confidence)
            else:
                st.error(f"{status} | Confidence: {confidence:.2f}%")

        st.image(image_np, caption="Processed Image", channels="RGB", use_container_width=True)


with tab2:
    camera_image = st.camera_input("Take a photo with your camera")
    if camera_image:
        pil_image = Image.open(camera_image).convert("RGB")
        image_np = prepare_image(pil_image)

        results = recognize_faces(image_np, embeddings, labels, conf_threshold=confidence_slider)

        for status, name, confidence, (top, right, bottom, left) in results:
            color = (0, 255, 0) if status.startswith("✅") else (255, 0, 0)
            cv2.rectangle(image_np, (left, top), (right, bottom), color, 2)
            cv2.putText(image_np, f"{status} {confidence:.1f}%", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if status.startswith("✅"):
                st.success(f"{status}: {name} | Confidence: {confidence:.2f}%")
                save_history(name, confidence)
            else:
                st.error(f"{status} | Confidence: {confidence:.2f}%")

        st.image(image_np, caption="Processed Image", channels="RGB", use_container_width=True)

# -----------------------------
# RECOGNITION HISTORY
# -----------------------------
st.markdown("---")
st.subheader("📜 Verification History")

if st.session_state.history:
    st.table(st.session_state.history)
else:
    st.info("No faces verified yet this session.")

