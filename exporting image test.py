import threading
import queue
import cv2
from deepface import DeepFace
import pyodbc
from io import BytesIO
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Queue to store frames for processing
frame_queue = queue.Queue(maxsize=5)

# Thread-safe result queue
result_queue = queue.Queue()

# Thread pool for managing face verification threads
executor = ThreadPoolExecutor(max_workers=2)  # Use limited threads to avoid overload

# Retrieve images from the database at startup
def get_images_from_db():
    connection = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;DATABASE=face_database;'
        'TRUSTED_CONNECTION=yes;'
    )
    cursor = connection.cursor()
    cursor.execute("SELECT person_name, image_data FROM FaceImages")

    images = []
    for row in cursor.fetchall():
        person_name = row[0]
        image_data = row[1]

        # Convert binary data to NumPy array (OpenCV BGR format)
        img = Image.open(BytesIO(image_data)).convert('RGB')
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to BGR

        images.append((person_name, img_bgr))

    cursor.close()
    connection.close()
    return images

# Preload images to avoid fetching from the database repeatedly
stored_images = get_images_from_db()

# Function to verify face in a separate thread
def verify_face_in_thread(frame):
    """Verifies the face and puts the result in the result queue."""
    for person_name, saved_image in stored_images:
        try:
            result = DeepFace.verify(
                frame, saved_image, detector_backend='opencv', model_name='VGG-Face'
            )
            if result["verified"]:
                result_queue.put(person_name)
                return  # Exit after finding the first match
        except Exception as e:
            print(f"Error verifying face: {e}")

    result_queue.put(None)  # No match found

# Main loop to capture video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize frame to reduce load (optional)
    resized_frame = cv2.resize(frame, (320, 240))

    # Process every nth frame (e.g., every 10th frame)
    if frame_queue.qsize() < 5:  # Add frame to queue if space is available
        frame_queue.put(resized_frame)

    # If a frame is available, start verification in a separate thread
    if not frame_queue.empty():
        next_frame = frame_queue.get()

        # Submit the face verification task to the thread pool
        executor.submit(verify_face_in_thread, next_frame)

    # Retrieve the result from the result queue, if available
    matched_person = None
    if not result_queue.empty():
        matched_person = result_queue.get()

    # Display the appropriate message
    if matched_person:
        cv2.putText(frame, f"Welcome {matched_person}!", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Welcome {matched_person}!")
    else:
        cv2.putText(frame, "No Match", (20, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the video frame
    cv2.imshow("Face Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
executor.shutdown()  # Gracefully shut down the thread pool
