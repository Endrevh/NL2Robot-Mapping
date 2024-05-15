from LLM import LargeLanguageModel
import mediapipe as mp
import pyrealsense2 as rs
import rtde_receive
import rtde_control
import os
import numpy as np
import cv2
import base64
import time
import ast
import json
from openai import OpenAI
from PIL import Image


client = OpenAI()

# Initialize camera
def initialize_camera(frequency):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, frequency)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, frequency)

    try:
        # Start the pipeline
        pipeline.start(config)
        print("RealSense camera initialized successfully.")
        return pipeline
    except Exception as e:
        print(f"Error initializing RealSense camera: {e}")
        return None

# Initialize robot
def initialize_robot(robot_ip_address):

    # Connect to robot
    rtde_c = rtde_control.RTDEControlInterface(robot_ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip_address)

    return rtde_c, rtde_r

# Load calibration data
def load_calibration_data(calibration_folder):
    with open(os.path.join(calibration_folder, "camera_matrix.txt"), 'r') as file:
        camera_matrix = np.loadtxt(file, delimiter=' ')
    with open(os.path.join(calibration_folder, "dist_coefficients.txt"), 'r') as file:
        dist_coeffs = np.loadtxt(file, delimiter=' ')
    with open(os.path.join(calibration_folder, "hand_eye_transformation.txt"), 'r') as file:
        hand_eye_transformation = np.loadtxt(file, delimiter=' ')

    return camera_matrix, dist_coeffs, hand_eye_transformation

def generate_tasks(language_model, pipeline):
    # Move end effector to initial_position

    # Capture image of worktable
    image = capture_image(pipeline)

    # Temporary save to file
    folder_path = "object detection/trash"
    file_name = "temp_image.png"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    cv2.imwrite(file_path, image)

    # Detect objects on workbench
    objects = detect_and_classify_objects_openai(file_path)
    #objects = detect_objects(image)

    # Print visible items to console
    print("Visible items:", objects)
    user_input = input("Enter prompt: ")
    
    prompt = "Visible items: " + str(objects) + ".  " + user_input
    #print("Prompt given to LLM: ", prompt)
    # Generate sequence of tasks using LLM
    sequence = language_model.generate_response(prompt)
    
    print("Generated sequence:", sequence)
    time.sleep(3)

    #First, represent the string as a list, then extract only the first element for each sublist
    nested_list = ast.literal_eval(sequence)
    action_sequence = [item[0] for item in nested_list]

    return action_sequence, objects


def capture_image(pipeline):
    # Implement code to capture image from current camera angle
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    # Convert image to numpy array and convert to RGB
    color_image = np.asanyarray(color_frame.get_data())
    #color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    return color_image  


def detect_objects(image):
    model_path = "object detection/models/efficientdet_lite2.tflite"
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        max_results=7,
        score_threshold = 0.10,
        running_mode=VisionRunningMode.IMAGE)

    with ObjectDetector.create_from_options(options) as detector:
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector.detect(rgb_frame)
        category_names = [detection.categories[0].category_name for detection in detection_result.detections]
        print("Detection results: ", category_names)
    return category_names

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

        # Draw bounding box
        start_point = (int(x), int(y))
        end_point = (int(x) + int(width), int(y) + int(height))
        color = (255, 0, 0)  # Blue color in BGR
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

        # Put label and confidence
        text = f"{label} ({confidence:.2f})"
        image = cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save annotated image
    annotated_image_path = os.path.splitext(image_path)[0] + '_annotated.png'
    cv2.imwrite(annotated_image_path, image)
    return annotated_image_path

def detect_and_classify_objects_openai(image_path):

    # Convert image from png to webp
    output_folder = "object detection/trash"
    os.makedirs(output_folder, exist_ok=True)
    image_path_webp = convert_png_to_webp(image_path, output_folder)

    # Encode image to base64
    image_base64 = encode_image_to_base64(image_path_webp)

    # Define the predefined categories
    predefined_categories = ["vehicle", "animal", "furniture", "electronics", "person", "hammer", "screwdriver", "wrench", "frisbee", "fork",
                            "knife", "toothbrush", "tie", "bottle", "scissors", "pliers", "saw"]

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

    # Create list from JSON reply


def execute_tasks(sequence, visible_objects, pipeline, rtde_c, rtde_r):
    repeat = False

    # Execute tasks in the sequence
    if sequence is None:
        print("No tasks to execute")
        return
    for task in sequence:
        action = task[0]
        action = str(action)
        # Use switch case to execute the corresponding action
        if action == "A":
            # Simulate picking up item
            speed = 0.4
            acc = 0.2
            #intermediate_pose = [-0.214, -0.506, 0.864, 0.559, -1.43, 2.02]
            #rtde_c.moveL(intermediate_pose, speed, acc, False)

            new_pose_upper = [-0.260, -0.908, 0.633, 0.64, -1.73, 1.85]
            #new_pose_lower = [-0.220, -0.564, 0.00205, 0.669, -1.84, 1.78]

            rtde_c.moveL(new_pose_upper, speed, acc, False)

        elif action == "B":
            # Give item to operator
            speed = 0.4
            acc = 0.2
            intermediate_pose = [-0.214, -0.506, 0.864, 0.559, -1.43, 2.02]
            rtde_c.moveL(intermediate_pose, speed, acc, False)
            new_pose = [-0.637, -0.554, 0.685, 0.218, 0.159, -5.50]            
            rtde_c.moveL(new_pose, speed, acc, False)

        elif action == "C":
            # Simulate equiping item
            speed = 0.1
            acc = 0.1
            intermediate_pose = [-0.214, -0.506, 0.864, 0.559, -1.43, 2.02]
            rtde_c.moveL(intermediate_pose, speed, acc, False)

            #new_pose_upper = [-0.260, -0.908, 0.633, 0.64, -1.73, 1.85]
            new_pose_lower = [-0.220, -0.564, 0.00205, 0.669, -1.84, 1.78]
            rtde_c.moveL(new_pose_lower, speed, acc, False)

        elif action == "D":
            # Move end effector to view workbench from new camera angle
            speed = 0.4
            acc = 0.2
            intermediate_pose = [-0.214, -0.506, 0.864, 0.559, -1.43, 2.02]
            rtde_c.moveL(intermediate_pose, speed, acc, False)
            new_pose = [-0.250, -0.815, 0.78, 0.59, -1.56, 1.96]
            rtde_c.moveL(new_pose, speed, acc, False)

        elif action == "E":
            image = capture_image(pipeline)

            # Temporary save to file
            folder_path = "object detection/trash"
            file_name = "temp_image.png"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path, file_name)
            cv2.imwrite(file_path, image)
            # Detect objects on workbench
            new_objects = detect_and_classify_objects_openai(file_path)
            # If new objects are not the same as old objects, a new call to generate_response is needed
            if new_objects != visible_objects:
                repeat = True
                return repeat

        else:
            print("Invalid action:", action)

    return repeat

# Initialize camera
calibration_folder = "camera_calibration"
camera_matrix, dist_coefficients, T_flange_eye = load_calibration_data(calibration_folder)
frequency = 30
pipeline = initialize_camera(frequency)

# initialize robot
robot_ip_address = "192.168.0.90"
rtde_c, rtde_r = initialize_robot(robot_ip_address)
print("Finished initializing robot")    

# Create LLM object and add system message
API_key = None # Replace with your API key if you have one
lang_model = "gpt-4o"
LLM = LargeLanguageModel(api_key=API_key, model=lang_model)
LLM.add_message("system", "You are my assistant and are in control of a robot with a camera mounted on the end-effector. The camera can be used to scan for items on a workbench. \\" +
                "The robot is able to perform the following actions: [A: Pick up item 'X' from the workbench. B: Give item 'X' to the me, the operator. C: Equip item 'X'. D: Move end-effector so that the upper workbench is visible to the camera. E: Capture image from current camera angle].\\" +
                "I, the operator, will tell you which items are visible on the workbench, and provide you with a task that I need done." + 
                "Your job is to create a sequence of actions from the list above which fullfils the task. \\" +
                "Your response (the sequence of tasks) should be a list in the format [task, object]. For example, ['A', 'hammer'] means pick up the hammer from the workbench. Some tasks are not object specific, for example, [D, ''] means move the end effector to view the workbench from a new camera angle. If no actions apply, return None. " +
                "Make sure that your response does not include anything else than the sequence of tasks on the correct format.")

# Move robot to initial position
initial_pose = [-0.248, -0.630, 0.343, 0.64, -1.85, 1.75]
speed = 0.1
acc = 0.1
rtde_c.moveL(initial_pose, speed, acc, False)

# Call the generate_tasks function with the user prompt
repeat = True
while repeat:
    # Generate sequence tasks
    sequence, objects = generate_tasks(LLM, pipeline)
    #print(sequence)
    # Execute tasks in sequence
    repeat = execute_tasks(sequence, objects, pipeline, rtde_c, rtde_r)


# Example usage of detect_and_classify_objects_openai
#image_path = "object detection/scene_images_tools/image_18.png"
#labels = detect_and_classify_objects_openai(image_path)
#print(labels)


