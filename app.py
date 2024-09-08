import cv2
import numpy as np
import face_recognition
import os
import streamlit as st

# Function to find encodings of known images
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

# Load images from the 'persons' folder
path = 'persons'
images = []
class_names = []
person_list = os.listdir(path)

for cl in person_list:
    cur_img = cv2.imread(f'{path}/{cl}')
    images.append(cur_img)
    class_names.append(os.path.splitext(cl)[0])

# Encode the known faces
encode_list_known = find_encodings(images)
st.write('Encoding Complete.')

# Project Overview Section
st.title("🎓 Masterclass Bootcamp Final Project")
st.header("Face Recognition Using OpenCV and Streamlit")
st.markdown("""
This is the **final project** of the Masterclass Bootcamp, where we developed a real-time face detection and recognition app using **OpenCV**, **Face Recognition** libraries, and **Streamlit** for the web interface. The system detects faces from a live webcam feed and compares them with the pre-loaded images.

### Key Features:
- 🧠 **Real-Time Face Detection**: Recognizes faces live from your webcam.
- 📸 **Pre-Trained Faces**: Matches faces against a set of pre-trained images.
- 🚀 **Simple Interface**: Start and stop the webcam feed with buttons.
- 💻 **Made with Aouf_Nihed**: Personalized project with my signature.

""")

# Streamlit App Layout
st.write("### Let's Get Started! 👇")
start_button = st.button("Start Webcam 🟢")
stop_button = st.button("Stop Webcam 🔴")
camera_frame = st.empty()

# Webcam Stream Logic
if start_button:
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Loop to capture webcam frames
    while True:
        success, img = cap.read()

        # If the frame is not successfully captured, skip
        if img is None:
            st.warning("Failed to capture the image.")
            continue

        # Resize the frame for faster processing
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(imgS)
        face_encodings = face_recognition.face_encodings(imgS, face_locations)

        # Process each face found in the frame
        for encode_face, face_loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(encode_list_known, encode_face)
            face_dis = face_recognition.face_distance(encode_list_known, encode_face)
            match_index = np.argmin(face_dis)

            if matches[match_index]:
                name = class_names[match_index].upper()
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Display the current frame in Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        camera_frame.image(img_rgb)

        # If Stop button is pressed, break the loop and release the webcam
        if stop_button:
            break

    # Release the webcam and close the Streamlit video frame
    cap.release()
    camera_frame.empty()

    # Show "Made with Aouf_Nihed" message
    st.write("### 👩‍💻 Made with ❤️ by **Aouf_Nihed**")

