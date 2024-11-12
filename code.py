import threading
import cv2
from deepface import DeepFace
import os

# Initialize the video capture from webcam
cap = cv2.VideoCapture(0)

# Set the resolution of the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize counter and face match flag
counter = 0
face_match = False

# Directory where reference photos are saved and checked
save_path = "C:/Users/varshithbr/Desktop/i gotcha/target_photos/"
os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

# Function to verify the current frame against all saved reference images
def check_face(frame):
    global face_match
    face_match = False  # Reset face match to False before checking

    try:
        # Iterate over all the saved photos in the folder
        for image_name in os.listdir(save_path):
            saved_image_path = os.path.join(save_path, image_name)
            saved_image = cv2.imread(saved_image_path)

            # Compare the current frame with each saved image
            if DeepFace.verify(frame, saved_image.copy())['verified']:
                face_match = True
                break  # Stop checking once a match is found

    except ValueError:
        face_match = False

# Main loop for video capture and face match checking
while True:
    ret, frame = cap.read()

    if ret:
        # Run face verification every 38 frames to avoid heavy computation on every frame
        if counter % 38 == 8:
            try:
                # Check the frame against all reference images in a separate thread
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass

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