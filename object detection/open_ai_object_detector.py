import os
import cv2
import base64
import json
from openai import OpenAI
from PIL import Image

client = OpenAI()

def draw_bounding_boxes(image_path, detections):
    # Load image
    image = cv2.imread(image_path)
    for detection in detections:
        box = detection['box']
        x = box['x']
        y = box['y']
        width = box['width']
        height = box['height']
        label = detection['label']
        confidence = detection['confidence']

        # Draw bounding box, use RGB (241,58,160) for screwdriver, RGB (58,162,93) for hammer and RGB (248,203,50) for wrench
        color = (255, 0, 0)  # Default color is blue (BGR)
        if label == "screwdriver":
            color = (160, 58, 241)
        elif label == "hammer":
            color = (93, 162, 58)
        elif label == "wrench":
            color = (50, 203, 248)
        else:
            color = (255, 0, 0)
        
        start_point = (int(x), int(y))
        end_point = (int(x) + int(width), int(y) + int(height))
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

        # Put label and confidence
        text = f"{label} ({confidence:.2f})"
        if int(y) > 20:
            image = cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else: 
            image = cv2.putText(image, text, (x, y + int(height) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save annotated image
    annotated_image_path = os.path.splitext(image_path)[0] + '_annotated.png'
    cv2.imwrite(annotated_image_path, image)
    return annotated_image_path

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def convert_png_to_webp(input_path, output_folder):
    # Ensure the file has a .png extension
    if not input_path.lower().endswith('.png'):
        print(f"Skipping {input_path}, as it is not a .png file.")
        return

    # Open the PNG image
    with Image.open(input_path) as img:
        # Extract the filename without the directory path
        filename = os.path.basename(input_path)
        # Change the extension to .webp
        output_filename = os.path.splitext(filename)[0] + '.webp'
        output_path = os.path.join(output_folder, output_filename)
        # Save the image in WEBP format
        img.save(output_path, 'webp')
        #print(f"Converted {filename} to {output_filename}")

    return output_path

def detect_and_classify_objects_openai(image_path, predefined_categories):

    #output_folder = "scene_images_tools"
    # Set output folder equal to the folder of the input image
    output_folder = os.path.dirname(image_path)
    os.makedirs(output_folder, exist_ok=True)

    # Convert the PNG image to WEBP format
    image_path_webp = convert_png_to_webp(image_path, output_folder)

    # Encode image to base64
    image_base64 = encode_image_to_base64(image_path_webp)

    # System message with instructions
    system_message = {
        "role": "system",
        "content": (
            f"You are an AI assistant trained to analyze images and provide output in JSON format. The user will provide an image in base64 format. "
            f"Your task is to detect objects in the image and classify them into one of the following categories: {', '.join(predefined_categories)}. "
            "Your output should be in JSON format. Each object should be formated with the following keys: 'box', 'label' and 'confidence'."
            "The 'box' key is a bounding box encapsulating the object and contains 'x', 'y', 'width', 'height'. The 'label' has to be one of the classes in the provided categories. The 'confidence' is the models certainty for the detection."
            "Make sure you stick to the format. No other text, info or data should be part of your reply, only detections."
        )
    }
    # User message with the base64 image
    user_message = {
        "role": "user",
        "content": [
                        {"type": "text", "text": f"Detect objects in this image according to the provided object categories and format. The image is 640x480 pixels for reference."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
    }

    # Call the OpenAI API's chat completions endpoint
    response = client.chat.completions.create(model="gpt-4o",
    messages=[system_message, user_message])

    # Extract and print the detected objects
    output = response.choices[0].message.content
    output_json = output.strip('`json\n') # Extract the JSON part from the string (remove the ```json and ``` at both ends)
    #print("JSON response: \n", output_json)
    detections = json.loads(output_json)
    draw_bounding_boxes(image_path_webp, detections)

    labels = [detection['label'] for detection in detections]
    return labels

# Define categories
predefined_categories = ["vehicle", "animal", "furniture", "electronics", "person", "hammer", "screwdriver", "wrench", "frisbee", "fork",
                            "knife", "toothbrush", "tie", "bottle", "scissors", "pliers", "saw"]

# Path to the image
image_path = "scene_images_tools/image_26.png"

# Detect and classify objects in the image
labels = detect_and_classify_objects_openai(image_path, predefined_categories)

print("Detected objects:", labels)
print("Annotated image saved in the same directory")