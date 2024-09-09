# **Real-Time Face Recognition System**  
## **Overview**  
This project is a real-time face recognition system designed to identify employees using their facial features. Built with **Streamlit**, **OpenCV**, and **face_recognition**, it captures webcam input, processes the face encodings, and matches them with a stored database of employee images. The project is ideal for applications like attendance tracking and secure access control.

## **Features**  
- Real-time face detection and recognition.
- Employee identification based on pre-encoded face data.
- Simple and intuitive user interface built with Streamlit.
- Scalable system for adding more employee data.

## **Tech Stack**
- **Streamlit**: Frontend web interface for displaying the face recognition system.
- **OpenCV**: Capture webcam video input and process face detection.
- **face_recognition**: Perform face encoding and comparison with stored data.
- **Pandas**: Manage and manipulate employee data.
- **OS**: File handling and directory management.

## **Installation**  
Follow these steps to set up and run the project locally.

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/AoufNihed/Masterclass_Bootcamp.git
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows: venv\Scripts\activate
    ```

3. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## **Directory Structure**  

- `/persons`: Folder where  images are stored.
- `app.py`: Main Streamlit application.
- `main.py`: Contains utility functions like face encoding and detection.

## **Usage**
Once the app is running, the webcam will be activated to capture real-time video. The system will compare detected faces with the stored employee face encodings and display the corresponding employee details if a match is found.

## **Requirements**
- Python 3.8 or higher.
- Webcam for real-time face detection.

## **Contributing**  
Feel free to fork this repository, make improvements, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
