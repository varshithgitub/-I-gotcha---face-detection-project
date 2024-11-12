import pyodbc
from PIL import Image
from io import BytesIO
import os

# Function to add an image to the database
def add_image_to_db(name, image_path):
    # Connect to the database
    connection = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;'
        'DATABASE=face_database;TRUSTED_CONNECTION=yes;'
    )
    cursor = connection.cursor()

    # Open the image file and convert it to binary
    with Image.open(image_path) as img:
        img_byte_arr = BytesIO()
        
        # Save the image in its original format to maintain quality
        if img.format in ['JPEG', 'JPG']:
            img.save(img_byte_arr, format='JPEG')  # Save as JPEG
        elif img.format == 'PNG':
            img.save(img_byte_arr, format='PNG')  # Save as PNG

        img_byte_arr = img_byte_arr.getvalue()

    # Insert image data into the database
    cursor.execute("INSERT INTO FaceImages (person_name, image_data) VALUES (?, ?)", (name, img_byte_arr))
    connection.commit()

    # Clean up
    cursor.close()
    connection.close()
    print(f"Image for {name} added to the database.")

# Specify the directory containing the images and add them to the database
def add_images_from_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image file types (case insensitive)
            image_path = os.path.join(directory, filename)
            person_name = os.path.splitext(filename)[0]  # Use filename without extension as the person name
            add_image_to_db(person_name, image_path)

# Specify the directory containing your images
images_directory = 'dump'  # Replace with your images directory
add_images_from_directory(images_directory)
