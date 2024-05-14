from LLM import *
import pyrealsense2 as rs
import rtde_receive
import rtde_control
import os
import numpy as np
import cv2

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

def generate_tasks(language_model, initial_position):
    # Move end effector to initial_position

    # Capture image of worktable
    image = capture_image()
    
    # Detect objects on workbench
    objects = detect_objects(image)
    
    # Print visible items to console
    print("Visible items:", objects)
    prompt = input("Enter prompt: ")

    # Generate sequence of tasks using LLM
    sequence = language_model.generate_response(prompt, objects)
    print("Generated sequence:", sequence)
    
    return sequence, objects
    
    
def capture_image():
    # Implement code to capture image from current camera angle
    # frames = pipeline.wait_for_frames()
    # color_frame = aligned_frames.get_color_frame()
    # Convert image to numpy array
    # color_image = np.asanyarray(color_frame.get_data())
    # color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    pass

def detect_objects(image):
    # Implement code to detect objects in the captured image
    pass

def execute_tasks(sequence, visible_objects):
    repeat = False

    # Execute tasks in the sequence
    if sequence is None:
        print("No tasks to execute")
        return
    for task in sequence:
        action, obj = task
        # Use switch case to execute the corresponding action
        #if action == "A":
            #pick_up_item(obj)
        #elif action == "B":
            #give_item_to_operator(obj)
        #elif action == "C":
            #equip_item(obj)
        #elif action == "D":
            #move_end_effector()
        #elif action == "E":
            # capture_image()

            # Detect objects on workbench
            # new_objects = detect_objects(color_image_rgb)
            # If new objects are not the same as old objects, a new call to generate_response is needed
            # if new_objects != visible_objects:
            #   repeat = True
            #   return repeat

        #else:
            #print("Invalid action:", action)
    pass

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
lang_model = None
LLM = LargeLanguageModel(api_key=API_key, model=lang_model)
LLM.add_message("system", "You are my assistant and are in control of a robot with a camera mounted on the end-effector. The camera can be used to scan for items on a workbench. \\" +
                "The robot is able to perform the following actions: [A: Pick up item 'X' from the workbench. B: Give item 'X' to the operator. C: Equip item 'X'. D: Move end-effector to view workbench from new camera angle. E: Capture image from current camera angle].\\" +
                "I, the operator, will tell you which items are visible on the workbench, and provide you with a task that I need done." + 
                "Your job is to create a sequence of actions from the list above which fullfils the task. \\" +
                "Your response (the sequence of tasks) should be a list in the format [task, object]. For example, [A, 'hammer'] means pick up the hammer from the workbench. Some tasks are not object specific, for example, [D, ''] means move the end effector to view the workbench from a new camera angle. If no actions apply, return None")

# Call the generate_tasks function with the user prompt
while repeat:
    # Generate sequence tasks
    sequence, objects = generate_tasks(LLM)
    # Execute tasks in sequence
    repeat = execute_tasks(sequence, objects, pipeline, rtde_c, rtde_r)


