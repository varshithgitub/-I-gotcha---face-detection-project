import threading
import queue
import cv2
from deepface import DeepFace
import pyodbc
from io import BytesIO
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time
import sys  # For system exit

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Queue to store frames for processing
frame_queue = queue.Queue(maxsize=10)

# Thread-safe result storage
matched_person = None
lock = threading.Lock()
prompt_shown = False
last_no_match_time = None

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=5)

# Graceful exit flag
exit_event = threading.Event()

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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
        person_name, image_data = row

        # Convert binary data to OpenCV image format
        img = Image.open(BytesIO(image_data)).convert('RGB')
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        images.append((person_name, img_bgr))

    cursor.close()
    connection.close()
    return images

stored_images = get_images_from_db()

def add_face_to_db(name, frame):
    connection = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;DATABASE=face_database;'
        'TRUSTED_CONNECTION=yes;'
    )
    cursor = connection.cursor()

    # Encode the frame as JPEG and store it in the database
    _, buffer = cv2.imencode('.jpg', frame)
    binary_data = buffer.tobytes()

    cursor.execute(
        "INSERT INTO FaceImages (person_name, image_data) VALUES (?, ?)",
        (name, binary_data)
    )
    connection.commit()
    cursor.close()
    connection.close()

    global stored_images
    stored_images = get_images_from_db()
    print(f"{name} added to the database successfully!")

def verify_face_in_thread(frame):
    global matched_person, last_no_match_time, prompt_shown
    match_found = False

    for person_name, saved_image in stored_images:
        result = DeepFace.verify(
            frame, saved_image, detector_backend='opencv', model_name='VGG-Face'
        )

        # Print the attributes of verification result
        print(f"Verification Result for {person_name}: {result}")

        # Check if the face matches with a high confidence level
        if result["verified"]:
            with lock:
                matched_person = person_name
                match_found = True
                prompt_shown = False
            print(f"Match found with {person_name}!")
            break  # Stop once a match is found

    if not match_found:
        with lock:
            matched_person = None
            if not prompt_shown:
                print("No match found in the database.")
                last_no_match_time = time.time()

def main_loop():
    global prompt_shown, matched_person

    while not exit_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        resized_frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        with lock:
            if len(faces) == 0:
                if matched_person:
                    matched_person = None
                    prompt_shown = False
                else:
                    matched_person = None

            if matched_person:
                cv2.putText(frame, f"Welcome {matched_person}!", (20, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Match Found", (20, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if last_no_match_time and (time.time() - last_no_match_time) > 15:
                    cv2.imshow("Face Recognition", frame)
                    

                    save_prompt = input("New face detected! Save this image? (yes/no): ")
                    if save_prompt.lower() == 'yes':
                        name = input("Enter name to save: ")
                        add_face_to_db(name, frame)

            if len(faces) > 0 and frame_queue.qsize() < 10:
                frame_queue.put(resized_frame)

            if not frame_queue.empty():
                next_frame = frame_queue.get()
                executor.submit(verify_face_in_thread, next_frame)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_event.set()
            break

def cleanup():
    """Ensure all resources are properly released."""
    print("Cleaning up resources...")
    exit_event.set()  # Signal threads to exit
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows
    executor.shutdown(wait=True)  # Wait for all threads to finish

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cleanup()
        sys.exit(0)  # Ensure the program exits cleanly
