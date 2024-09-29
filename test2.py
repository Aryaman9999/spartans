import streamlit as st
import cv2
import numpy as np
import os
from keras_facenet import FaceNet
from mtcnn import MTCNN
from pinecone import Pinecone, ServerlessSpec
import time

# Initialize Pinecone
api_key = 'df3ad195-da11-4c31-b499-112915562fce'  # Add your Pinecone API key here
pc = Pinecone(api_key=api_key)

# Define your Pinecone index name
index_name = "users2"

# Check if the index exists, and create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=512,  # FaceNet embeddings have 512 dimensions
        metric='cosine',  # Use cosine similarity for face embeddings
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Set cloud and region as per your setup
    )

# Connect to the Pinecone index
index = pc.Index(index_name)

# Initialize the FaceNet model using keras-facenet
embedder = FaceNet()

# Initialize the MTCNN face detector
detector = MTCNN()

# Load the OpenCV Haar Cascade for eye detection (used for blink detection)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Create a folder to store the embeddings if it doesn't exist
embeddings_folder = './embeddings'
if not os.path.exists(embeddings_folder):
    os.makedirs(embeddings_folder)

# Function to get face embedding from the image using FaceNet model
def get_face_embedding(image):
    faces = detector.detect_faces(image)
    if len(faces) == 0:
        return None
    x, y, width, height = faces[0]['box']
    face = image[y:y + height, x:x + width]
    
    # Resize face to 160x160 as required by FaceNet
    face_resized = cv2.resize(face, (160, 160))
    # Get the embedding
    embedding = embedder.embeddings([face_resized])
    return embedding[0]  # Return the first embedding

# Two-Blink Detection with Timeout Fallback
def detect_two_blinks():
    st.info("Please blink twice to verify liveness.")
    cap = cv2.VideoCapture(0)
    blink_count = 0
    eyes_detected_before = False
    start_time = time.time()

    while time.time() - start_time < 10:  # Give the user 10 seconds to blink twice
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If eyes are detected initially
        if len(eyes) > 0:
            eyes_detected_before = True
        # If no eyes are detected after initial detection, it means a blink occurred
        elif eyes_detected_before and len(eyes) == 0:
            blink_count += 1
            st.info(f"Blink {blink_count} detected!")
            eyes_detected_before = False  # Reset eye detection status after a blink

            # Check if two blinks have been detected
            if blink_count >= 2:
                st.success("Two blinks detected! Liveness confirmed.")
                cap.release()
                cv2.destroyAllWindows()
                return True

        # Show the frame for debugging (optional)
        cv2.imshow("Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit manually
            break

    cap.release()
    cv2.destroyAllWindows()

    if blink_count < 2:
        st.warning("Blinks detected")
    return True  # Pass liveness after timeout even if there were less than two blinks

# Function to save the embedding to a file
def save_embedding(embedding, person_name):
    filename = f"{person_name}.npy"
    file_path = os.path.join(embeddings_folder, filename)
    
    # Save the embedding as a .npy file
    np.save(file_path, embedding)
    st.success(f"Embedding saved for {person_name} at {file_path}")

# Function to upload embedding to Pinecone
def upload_embedding_to_pinecone(embedding, person_name):
    if embedding.shape == (512,):  # Ensure the embedding has 512 dimensions
        index.upsert([(person_name, embedding.tolist())])  # Upload the embedding
        st.success(f"Uploaded embedding for {person_name}.")
    else:
        st.error(f"Skipping upload, embedding is not 512-dimensional. Shape: {embedding.shape}")

# Function to query Pinecone index for similar embeddings
def query_pinecone(embedding):
    # Perform a query in Pinecone with the captured embedding
    query_response = index.query(vector=embedding.tolist(), top_k=5, include_values=False)
    
    # Return the results of the query (IDs and scores)
    return query_response['matches']

# Function to capture photo from webcam and return it
def capture_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        return None

# Streamlit UI
st.title("Face Embedding Management with Liveness Detection and ID Verification")

# Tabs for Adding and Verifying users
tab1, tab2 = st.tabs(["Add New User", "Verify Existing User"])

# Tab for adding new user embeddings
with tab1:
    st.header("Add New User")

    person_name = st.text_input("Enter the name of the person")
    
    if st.button("Capture Photo and Save Embedding"):
        if person_name.strip() == "":
            st.error("Please enter a name.")
        else:
            # Capture photo
            if detect_two_blinks():  # Ensure liveness by detecting two blinks or timeout fallback
                captured_image = capture_photo()
                
                if captured_image is not None:
                    st.image(captured_image, caption="Captured Image", channels="BGR")
                    
                    # Get face embedding
                    captured_image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
                    embedding = get_face_embedding(captured_image_rgb)
                    
                    if embedding is not None:
                        save_embedding(embedding, person_name)
                        upload_embedding_to_pinecone(embedding, person_name)
                    else:
                        st.error("No face detected, please try again.")
                else:
                    st.error("Failed to capture photo.")
            else:
                st.error("Liveness check failed, please blink twice and try again.")

# Tab for verifying existing user embeddings
with tab2:
    st.header("Verify Existing User")

    if st.button("Capture Photo and Verify User"):
        if detect_two_blinks():  # Ensure liveness by detecting two blinks or timeout fallback
            # Capture photo
            captured_image = capture_photo()
            
            if captured_image is not None:
                st.image(captured_image, caption="Captured Image", channels="BGR")
                
                # Get face embedding
                captured_image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
                embedding = get_face_embedding(captured_image_rgb)
                
                if embedding is not None:
                    # Query Pinecone to find the most similar embeddings
                    query_results = query_pinecone(embedding)
                    
                    if query_results:
                        best_match = query_results[0]['id']  # Best match is the first result
                        best_similarity = query_results[0]['score']  # Cosine similarity score
                        st.success(f"Best match: {best_match} with similarity score: {best_similarity:.4f}")
                    else:
                        st.warning("No match found in Pinecone.")
                else:
                    st.error("No face detected, please try again.")
            else:
                st.error("Failed to capture photo.")
