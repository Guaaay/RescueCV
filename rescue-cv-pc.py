import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import threading
import time


# Model available to download here: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
model_path = "pose_landmarker_lite_model.task"

video_source = 0

num_poses = 2
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5

danger_red = (0,0,255)
danger_green = (0,255,0)
danger_yellow = (22,234,231)


'''
FONT_HERSHEY_SIMPLEX        = 0, //!< normal size sans-serif font
    FONT_HERSHEY_PLAIN          = 1, //!< small size sans-serif font
    FONT_HERSHEY_DUPLEX         = 2, //!< normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)
    FONT_HERSHEY_COMPLEX        = 3, //!< normal size serif font
    FONT_HERSHEY_TRIPLEX        = 4, //!< normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
    FONT_HERSHEY_COMPLEX_SMALL  = 5, //!< smaller version of FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_SCRIPT_SIMPLEX = 6, //!< hand-writing style font
    FONT_HERSHEY_SCRIPT_COMPLEX = 7, //!< more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_ITALIC                 = 16,
'''

def putText(text, xcord, ycord, font):
    global frame
    cv2.putText(frame,  
                text,  
                (xcord, ycord),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
'''
def putRectangle(frame, centre_x, size, centre_y, ):
    cv2.rectangle(frame, start_point, end_point, 'red', 1)
    '''
def putScope(frame, centre_x, centre_y, size, scope_size, color = danger_green):
    centre_topleft = (centre_x - size, centre_y + size)
    centre_bottomright = (centre_x + size, centre_y - size)
    
    #top point
    cv2.rectangle(frame, (centre_topleft[0],centre_topleft[1] + 100*scope_size), (centre_bottomright[0], centre_bottomright[1] + scope_size), color, -1)
     
    #right point
    cv2.rectangle(frame, (centre_topleft[0] + scope_size,centre_topleft[1]) , (centre_bottomright[0]+100*scope_size, centre_bottomright[1]), color, -1)
     
    #left point
    cv2.rectangle(frame, (centre_topleft[0] -  scope_size,centre_topleft[1]) , (centre_bottomright[0]-100*scope_size, centre_bottomright[1]) ,color, -1)
    
    #down point
    cv2.rectangle(frame, (centre_topleft[0], centre_topleft[1]-100*scope_size) , (centre_bottomright[0] , centre_bottomright[1] - scope_size), color, -1)



def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


to_window = None
last_timestamp_ms = 0

def z_estimation(pose_landmarks):
    mean_dist = 0
    count = 0
    ref_landmark = pose_landmarks[0]
    for landmark in pose_landmarks:
        count += 1
        mean_dist += np.sqrt((landmark.x - ref_landmark.x)**2 + (landmark.y - ref_landmark.y)**2)
    mean_dist /= count
    return mean_dist

def are_joints_visible(left_hip,right_hip, left_shoulder,right_shoulder):
    threshold = 0.3
    is_visible_left_hip = left_hip.visibility > threshold
    is_visible_right_hip = right_hip.visibility > threshold
    is_visible_left_shoulder = left_shoulder.visibility > threshold
    is_visible_right_shoulder = right_shoulder.visibility > threshold
    return is_visible_left_hip and is_visible_right_hip and is_visible_left_shoulder and is_visible_right_shoulder


def check_vertical(lower_joint, upper_joint, pose_landmarks):
    threshold = 0.3*z_estimation(pose_landmarks)
    
    x_distance = abs(lower_joint.x - upper_joint.x)
    #print("distance",x_distance)
    if x_distance < threshold:
        return True
    return False


def is_standing(pose_landmarks) -> bool:
    left_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    left_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    #Check for joint visibility
    
    if(not are_joints_visible(left_hip,right_hip, left_shoulder,right_shoulder)):
        return False
    
    if(check_vertical(left_hip,left_shoulder,pose_landmarks) and check_vertical(right_hip,right_shoulder,pose_landmarks)):
        return True
    
    #print(right_shoulder)
    #print(left_shoulder)

    return False

#################################
def is_joint_visible(joint, threshold=0.3):
    return joint.visibility > threshold

def check_y_distance(lower_joint, upper_joint, pose_landmarks, threshold):
    threshold = threshold*z_estimation(pose_landmarks)
    y_distance = abs(lower_joint.y - upper_joint.y)
    if y_distance < threshold:
        return True
    return False

def is_lying_down(pose_landmarks):
    head = pose_landmarks[mp.solutions.pose.PoseLandmark.NOSE]
    left_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    if is_joint_visible(head) and is_joint_visible(left_hip) and is_joint_visible(right_hip):
        return check_y_distance(left_hip, head, pose_landmarks, 3) and check_y_distance(right_hip, head, pose_landmarks, 0.6)


def is_crouching(pose_landmarks):
    left_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    left_knee = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
    right_knee = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
    if is_joint_visible(left_hip) and is_joint_visible(left_knee):
        return check_y_distance(left_hip, left_knee, pose_landmarks, 0.4)
    
    if is_joint_visible(right_hip) and is_joint_visible(right_knee):
        return check_y_distance(right_hip, right_knee, pose_landmarks, 0.4)

    return False
#########################################

def get_center(pose_landmarks):
    left_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    left_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    if(not are_joints_visible(left_hip,right_hip, left_shoulder,right_shoulder)):
        return (0,0)
    x = (left_hip.x + right_hip.x + left_shoulder.x + right_shoulder.x)/4
    y = (left_hip.y + right_hip.y + left_shoulder.y + right_shoulder.y)/4
    return (x,y)
    

def classify_pose(pose_landmarks: landmark_pb2.NormalizedLandmarkList) -> str:
    # Classify the pose based on the pose landmarks
    # print("pose landmarks: {}".format(pose_landmarks))
    standing = is_standing(pose_landmarks)
    crouching = is_crouching(pose_landmarks)
    lying = is_lying_down(pose_landmarks)
    if(lying):
        return "PELIGRO MUY ALTO"
    elif(crouching):
        return "PELIGRO ALTO"
    else:
        return "PELIGRO"
    return "UNKNOWN"

def denormalize(tup, height, width):
    return (int(tup[0]*width), int(tup[1]*height))




def print_result(detection_result: vision.PoseLandmarkerResult, output_image: mp.Image,
                 timestamp_ms: int):
    global to_window
    global last_timestamp_ms
    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms
    # print("pose landmarker result: {}".format(detection_result))
    #print(detection_result)
    height, width, channels = output_image.numpy_view().shape 
    center = (0,0)
    z_est = 1.0
    pose = "UNKNOWN"
    pose_list = []
    if(len(detection_result.pose_landmarks) > 0):
        for pose_landmarks in detection_result.pose_landmarks:
            pose = classify_pose(pose_landmarks)
            center = get_center(pose_landmarks)
            center = denormalize(center, height, width)
            z_est = z_estimation(pose_landmarks)
            pose_list.append([pose, center, z_est])
        


    to_window = cv2.cvtColor(
        output_image.numpy_view(), cv2.COLOR_RGB2BGR)
    # to_window = cv2.cvtColor(
    #     draw_landmarks_on_image(output_image.numpy_view(), detection_result), cv2.COLOR_RGB2BGR)
    max_level = 0
    for pose, center, z_est in pose_list:
        if(center[0] > 0 and center[1] > 0):
            
            color = danger_green
            if(pose == "PELIGRO"):
                level = 0
                color = danger_green
            elif(pose == "PELIGRO MUY ALTO"):
                level = 2
                if(level > max_level):
                    max_level = level
                
                color = danger_red
            elif(pose == "PELIGRO ALTO"):
                level = 1
                color = danger_yellow

            if(level > max_level):
                    max_level = level
            
            putScope(to_window, center[0], center[1], int(50*(z_est*0.1)), int(400*(z_est)), color)
            to_window = cv2.putText(to_window, pose, (center[0]-int(400*z_est), center[1]-int(300*z_est)), cv2.FONT_HERSHEY_PLAIN, 100*(z_est*0.1), color, 2, cv2.LINE_4)

            
            
    


def main():
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
        result_callback=print_result
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        # Use OpenCV’s VideoCapture to start capturing from the webcam.
        cap = cv2.VideoCapture(video_source)

        # Create a loop to read the latest frame from the camera using VideoCapture#read()
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Image capture failed.")
                break

            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            if to_window is not None:

                window_name = "RescueCV"
                cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    
                # Set the window to fullscreen
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
                cv2.imshow(window_name, to_window)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()