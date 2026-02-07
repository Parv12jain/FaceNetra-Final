ADMIN_PIN = "1234"  # change this later
import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image

from database import insert_face
from recognition import recognize_face
from attendance import mark_attendance

from mtcnn import MTCNN
from keras_facenet import FaceNet

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False



# ================= MODEL CACHING (CRITICAL FIX) =================
@st.cache_resource
def load_models():
    detector = MTCNN()
    embedder = FaceNet()
    return detector, embedder

detector, embedder = load_models()
# ===============================================================


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FaceNetra",
    page_icon="👁️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #38bdf8;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}
.card {
    background-color: #020617;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(56,189,248,0.15);
}
.success-box {
    background-color: #022c22;
    padding: 15px;
    border-radius: 10px;
    color: #34d399;
}
.error-box {
    background-color: #3f1d1d;
    padding: 15px;
    border-radius: 10px;
    color: #f87171;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">👁️ FaceNetra</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Face Recognition & Attendance</div>', unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Webcam Attendance",
    "📁 Upload Image",
    "➕ Register via Camera",
    "🪪 Register via Photo Upload",
    "🛠 Admin Panel"
])


# ================= TAB 1: WEBCAM (AUTO ATTENDANCE) =================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Webcam Capture")

    camera_image = st.camera_input("Capture face and mark attendance")

    if camera_image:
        image_bytes = np.frombuffer(camera_image.getvalue(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        boxed_image, result = recognize_face(image)
        st.image(boxed_image, caption="Detected Face", width=700)

        name = result["name"]
        confidence = result["confidence"]

        if name != "Unknown":
            marked = mark_attendance(name, confidence)
            if marked:
                st.markdown(
                    f'<div class="success-box">✅ Attendance marked for {name}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("⏱ Attendance already marked recently")
        else:
            st.markdown(
                '<div class="error-box">❌ Unknown face – attendance not allowed</div>',
                unsafe_allow_html=True
            )

        st.write("**Result:**", result)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 2: IMAGE UPLOAD =================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Face Image")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        boxed_image, result = recognize_face(image_np)
        st.image(boxed_image, caption="Detected Face", width=700)

        if result["name"] != "Unknown":
            st.markdown(
                f'<div class="success-box">Recognized: {result["name"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="error-box">Unknown Face</div>',
                unsafe_allow_html=True
            )

        st.write("**Result:**", result)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 3: REGISTER NEW USER =================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Register New Face")

    person_name = st.text_input("Enter person's name")
    register_image = st.camera_input("Capture face for registration")

    if register_image and person_name:
        image_bytes = np.frombuffer(register_image.getvalue(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(rgb)

        if not faces:
            st.error("No face detected. Try again.")
        else:
            face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)

            face_img = rgb[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (160, 160))
            face_img = face_img.astype("float32")

            embedding = embedder.embeddings([face_img])[0]
            embedding /= np.linalg.norm(embedding)

            insert_face(person_name, embedding)
            st.success(f"✅ {person_name} registered successfully")

    st.markdown('</div>', unsafe_allow_html=True)


# ================= TAB 4: PASSPORT PHOTO REGISTRATION =================
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Register via Passport-Size Photo")

    person_name = st.text_input("Enter full name (as per ID)", key="passport_name")
    uploaded_photo = st.file_uploader(
        "Upload a clear passport-size photo",
        type=["jpg", "jpeg", "png"],
        key="passport_upload"
    )

    if uploaded_photo and person_name:
        image = Image.open(uploaded_photo).convert("RGB")
        image_np = np.array(image)

        h, w, _ = image_np.shape

        # ❌ Image too small
        if w < 300 or h < 300:
            st.error("❌ Image resolution too low. Upload a clearer photo.")
            st.stop()

        rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        faces = detector.detect_faces(rgb)

        # ❌ No face
        if len(faces) == 0:
            st.error("❌ No face detected. Upload a clear front-facing photo.")
            st.stop()

        # ❌ Multiple faces
        if len(faces) > 1:
            st.error("❌ Multiple faces detected. Upload only your own photo.")
            st.stop()

        face = faces[0]
        x, y, w_face, h_face = face["box"]
        x, y = max(0, x), max(0, y)

        face_area = w_face * h_face
        image_area = h * w

        # ❌ Face too small
        if face_area / image_area < 0.08:

            st.error("❌ Face too small. Upload a closer passport-size photo.")
            st.stop()

        # Crop + embed
        face_img = rgb[y:y+h_face, x:x+w_face]
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype("float32")

        embedding = embedder.embeddings([face_img])[0]
        embedding /= np.linalg.norm(embedding)

        insert_face(person_name, embedding)

        st.success(f"✅ {person_name} registered successfully via passport photo")
        st.info("System will reload to activate recognition")
        st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ================= TAB 5: ADMIN VIEW=================

from database import get_registered_users, delete_user

with tab5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔐 Admin Panel")

    # 🔒 If not authenticated
    if not st.session_state.admin_authenticated:
        pin = st.text_input("Enter Admin PIN", type="password")

        if st.button("Login"):
            if pin == ADMIN_PIN:
                st.session_state.admin_authenticated = True
                st.success("✅ Admin authenticated")
                st.rerun()

            else:
                st.error("❌ Incorrect PIN")

    # ✅ If authenticated
    else:
        st.success("Welcome, Admin")

        from database import get_registered_users

        users = get_registered_users()

        if users:
            st.subheader("📋 Registered Users")
            st.table(users)
        else:
            st.info("No registered users found")

        if st.button("Logout"):
            st.session_state.admin_authenticated = False
            st.rerun()


    st.markdown('</div>', unsafe_allow_html=True)



# ================= ATTENDANCE REPORT =================
st.divider()
st.subheader("📊 Attendance Report")

try:
    df = pd.read_csv("attendance.csv")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Attendance CSV", csv, "attendance.csv", "text/csv")
except:
    st.info("No attendance records yet")


# import streamlit as st
# import pandas as pd
# import cv2
# import numpy as np
# from PIL import Image

# from database import insert_face
# from recognition import recognize_face
# from attendance import mark_attendance

# from mtcnn import MTCNN
# from keras_facenet import FaceNet


# # ================= MODEL CACHING =================
# @st.cache_resource
# def load_models():
#     detector = MTCNN()
#     embedder = FaceNet()
#     return detector, embedder

# detector, embedder = load_models()
# # =================================================


# # ================= CAMERA STATE =================
# if "cap" not in st.session_state:
#     st.session_state.cap = None

# if "camera_on" not in st.session_state:
#     st.session_state.camera_on = False
# # =================================================


# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(
#     page_title="FaceNetra",
#     page_icon="👁️",
#     layout="wide"
# )

# # ---------------- CUSTOM CSS ----------------
# st.markdown("""
# <style>
# body {
#     background-color: #0f172a;
# }
# .main-title {
#     text-align: center;
#     font-size: 42px;
#     font-weight: 700;
#     color: #38bdf8;
# }
# .subtitle {
#     text-align: center;
#     color: #cbd5e1;
#     font-size: 18px;
#     margin-bottom: 30px;
# }
# .card {
#     background-color: #020617;
#     padding: 25px;
#     border-radius: 15px;
#     box-shadow: 0px 0px 20px rgba(56,189,248,0.15);
# }
# .success-box {
#     background-color: #022c22;
#     padding: 15px;
#     border-radius: 10px;
#     color: #34d399;
# }
# .error-box {
#     background-color: #3f1d1d;
#     padding: 15px;
#     border-radius: 10px;
#     color: #f87171;
# }
# </style>
# """, unsafe_allow_html=True)

# # ---------------- HEADER ----------------
# st.markdown('<div class="main-title">👁️ FaceNetra</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitle">AI-Powered Face Recognition & Attendance</div>', unsafe_allow_html=True)

# # ---------------- TABS ----------------
# tab1, tab2, tab3 = st.tabs([
#     "📷 Webcam Attendance",
#     "📁 Upload Image",
#     "➕ Register New User"
# ])


# # ================= TAB 1: WEBCAM (OPENCV STREAM) =================
# with tab1:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Live Webcam Capture")

#     col1, col2 = st.columns(2)

#     with col1:
#         if st.button("▶ Start Camera"):
#             st.session_state.cap = cv2.VideoCapture(0)  # change to 1/2 for phone cam
#             st.session_state.camera_on = True

#     with col2:
#         if st.button("⏹ Stop Camera"):
#             if st.session_state.cap:
#                 st.session_state.cap.release()
#             st.session_state.camera_on = False

#     frame_placeholder = st.empty()
#     result_placeholder = st.empty()

#     if st.session_state.camera_on and st.session_state.cap:
#         ret, frame = st.session_state.cap.read()

#         if ret:
#             boxed_image, result = recognize_face(frame)

#             frame_placeholder.image(
#                 cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB),
#                 caption="Detected Face",
#                 width=700
#             )

#             name = result["name"]
#             confidence = result["confidence"]

#             if name != "Unknown":
#                 marked = mark_attendance(name, confidence)
#                 if marked:
#                     result_placeholder.markdown(
#                         f'<div class="success-box">✅ Attendance marked for {name}</div>',
#                         unsafe_allow_html=True
#                     )
#                 else:
#                     result_placeholder.info("⏱ Attendance already marked recently")
#             else:
#                 result_placeholder.markdown(
#                     '<div class="error-box">❌ Unknown face – attendance not allowed</div>',
#                     unsafe_allow_html=True
#                 )

#             result_placeholder.markdown(f"**Result:** {result}")


#     st.markdown('</div>', unsafe_allow_html=True)


# # ================= TAB 2: IMAGE UPLOAD =================
# with tab2:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Upload Face Image")

#     uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

#     if uploaded_file:
#         image = Image.open(uploaded_file).convert("RGB")
#         image_np = np.array(image)

#         boxed_image, result = recognize_face(image_np)
#         st.image(boxed_image, caption="Detected Face", width=700)

#         if result["name"] != "Unknown":
#             st.markdown(
#                 f'<div class="success-box">Recognized: {result["name"]}</div>',
#                 unsafe_allow_html=True
#             )
#         else:
#             st.markdown(
#                 '<div class="error-box">Unknown Face</div>',
#                 unsafe_allow_html=True
#             )

#         st.write("**Result:**", result)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ================= TAB 3: REGISTER NEW USER (PHONE CAM) =================
# with tab3:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Register New Face")

#     person_name = st.text_input("Enter person's name")

#     if st.button("📸 Capture Face from Camera") and person_name:
#         if not st.session_state.cap:
#             st.session_state.cap = cv2.VideoCapture(0)

#         ret, frame = st.session_state.cap.read()

#         if ret:
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             faces = detector.detect_faces(rgb)

#             if not faces:
#                 st.error("No face detected. Try again.")
#             else:
#                 face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
#                 x, y, w, h = face["box"]
#                 x, y = max(0, x), max(0, y)

#                 face_img = rgb[y:y+h, x:x+w]
#                 face_img = cv2.resize(face_img, (160, 160))
#                 face_img = face_img.astype("float32")

#                 embedding = embedder.embeddings([face_img])[0]
#                 embedding /= np.linalg.norm(embedding)

#                 insert_face(person_name, embedding)
#                 st.success(f"✅ {person_name} registered successfully")

#     st.markdown('</div>', unsafe_allow_html=True)


# # ================= ATTENDANCE REPORT =================
# st.divider()
# st.subheader("📊 Attendance Report")

# try:
#     df = pd.read_csv("attendance.csv")
#     st.dataframe(df, use_container_width=True)

#     csv = df.to_csv(index=False).encode("utf-8")
#     st.download_button(
#         "⬇️ Download Attendance CSV",
#         csv,
#         "attendance.csv",
#         "text/csv"
#     )
# except:
#     st.info("No attendance records yet")
