import cv2
import numpy as np
import face_recognition
import os

#encodings images
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

# Load images 
path = 'persons'
images = []
class_names = []
person_list = os.listdir(path)

for cl in person_list:
    cur_img = cv2.imread(f'{path}/{cl}')
    images.append(cur_img)
    class_names.append(os.path.splitext(cl)[0])

# Encode 
encode_list_known = find_encodings(images)
print('Encoding Complete.')

# Open webcam 
cap = cv2.VideoCapture(0) 

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    success, img = cap.read()


    if img is None:
        print("Failed to capture the image.")
        continue

    
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)


    face_locations = face_recognition.face_locations(imgS)
    face_encodings = face_recognition.face_encodings(imgS, face_locations)

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

    # Display  image
    cv2.imshow('Face Recognition', img)

    # Break loop 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
