import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Model available to download here: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
model_path = "pose_landmarker_lite.task"

video_source = 0

num_poses = 1
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5


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
def putScope(frame, centre_x, centre_y, size, scope_size):
    size = int(size)
    scope_size = int(scope_size)
    centre_topleft = (centre_x - size, centre_y + 2*size)
    centre_bottomright = (centre_x + size, centre_y - 2*size)
    #top point
    cv2.rectangle(frame, (centre_topleft[0],centre_topleft[1] + scope_size), (centre_bottomright[0], centre_bottomright[1] + scope_size), (0,0,0), -1)
    #right point
    cv2.rectangle(frame, (centre_topleft[0] + scope_size,centre_topleft[1]) , (centre_bottomright[0]+scope_size, centre_bottomright[1]), (0,0,0), -1)
    #left point
    cv2.rectangle(frame, (centre_topleft[0] -  scope_size,centre_topleft[1]) , (centre_bottomright[0]-scope_size, centre_bottomright[1]) ,(0,0,0), -1)
    #down point
    cv2.rectangle(frame, (centre_topleft[0], centre_topleft[1]-scope_size) , (centre_bottomright[0] , centre_bottomright[1] - scope_size), (0,0,0), -1)



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
    if(is_standing(pose_landmarks)):
        return "STANDING"
    else:
        return "LYING DOWN"

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
    center = (0,0)
    z_est = 1.0
    pose = "UNKNOWN"
    if(len(detection_result.pose_landmarks) > 0):
        pose = classify_pose(detection_result.pose_landmarks[0])
        center = get_center(detection_result.pose_landmarks[0])
        z_est = z_estimation(detection_result.pose_landmarks[0])

    height, width, channels = output_image.numpy_view().shape 
    

    center = denormalize(center, height, width)

    print(center)

    
    to_window = cv2.cvtColor(
        draw_landmarks_on_image(output_image.numpy_view(), detection_result), cv2.COLOR_RGB2BGR)
    
    if(center[0] > 0 and center[1] > 0):
        putScope(to_window, center[0], center[1], 10*z_est, 200*z_est)
        to_window = cv2.putText(to_window, pose, (center[0]-int(100*z_est), center[1]), cv2.FONT_HERSHEY_SIMPLEX, 10*z_est, (0, 255, 255), 2, cv2.LINE_4)



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
                cv2.imshow("MediaPipe Pose Landmark", to_window)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()