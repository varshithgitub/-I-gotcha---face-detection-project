import threading
import cv2
import pyodbc
from deepface import DeepFace
import numpy as np
import json

# Initialize the video capture from webcam
cap = cv2.VideoCapture(0)

# Set the resolution of the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize counter and face match flag
counter = 0
face_match = False

# MS SQL Server connection details using Windows Authentication
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=localhost;'  # Replace with your server name
    'DATABASE=face_database;'  # Replace with your database name
    'Trusted_Connection=yes;'  # Use Windows Authentication
)

# Function to insert face embeddings into MS SQL database
def save_embedding_to_db(embedding):
    # Convert embedding (list) to string for storage
    embedding_str = json.dumps(embedding)  # No need for .tolist()

    # Insert the face embedding into the database
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO ReferenceEmbeddings (Embedding) VALUES (?)", 
        (embedding_str,)
    )
    conn.commit()


# Function to load face embeddings from the database
def load_embeddings_from_db():
    cursor = conn.cursor()
    cursor.execute("SELECT Embedding FROM ReferenceEmbeddings")
    
    saved_embeddings = []
    for row in cursor.fetchall():
        # Convert the string data back to a numpy array
        embedding = np.array(json.loads(row[0]))
        saved_embeddings.append(embedding)
    
    return saved_embeddings

# Function to extract the face embedding from the current frame
def get_face_embedding(frame):
    try:
        # Use DeepFace to extract the face embedding
        embedding = DeepFace.represent(frame, model_name='VGG-Face')[0]["embedding"]
        return embedding
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None

# Function to verify the current frame's face embedding against all saved embeddings
def check_face(frame):
    global face_match
    face_match = False  # Reset face match to False before checking

    try:
        # Get face embedding of the current frame
        current_embedding = get_face_embedding(frame)
        if current_embedding is None:
            return

        # Load saved embeddings from the database
        saved_embeddings = load_embeddings_from_db()

        # Compare the current embedding with each saved embedding
        for saved_embedding in saved_embeddings:
            distance = np.linalg.norm(np.array(saved_embedding) - np.array(current_embedding))
            
            # Threshold for face match (adjust as necessary)
            if distance < 0.6:  # Typically, 0.6 is a good threshold for VGG-Face
                face_match = True
                break  # Stop checking once a match is found

    except ValueError as e:
        print(f"Error during face verification: {e}")
        face_match = False

# Main loop for video capture and face match checking
while True:
    ret, frame = cap.read()

    if ret:
        # Run face verification every 38 frames to avoid heavy computation on every frame
        if counter % 38 == 8:
            try:
                # Check the frame against all reference embeddings in a separate thread
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass

        # Save the face embedding into the database at certain intervals (e.g., every 100 frames)
        if counter % 100 == 0:
            face_embedding = get_face_embedding(frame)
            if face_embedding is not None:
                save_embedding_to_db(face_embedding)

        counter += 1

        # If a face match is found, display "MATCH", otherwise display "NO MATCH"
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Show the video stream
        cv2.imshow("video", frame)

    # Exit when 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
conn.close()  # Close the database connection
