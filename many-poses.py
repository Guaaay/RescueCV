import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Model available to download here: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
model_path = "pose_landmarker_lite.task"

video_source = 0

num_poses = 2
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5


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

def are_joints_visible(left_hip,right_hip, left_shoulder,right_shoulder):
    threshold = 0.3
    is_visible_left_hip = left_hip.visibility > threshold
    is_visible_right_hip = right_hip.visibility > threshold
    is_visible_left_shoulder = left_shoulder.visibility > threshold
    is_visible_right_shoulder = right_shoulder.visibility > threshold
    return is_visible_left_hip and is_visible_right_hip and is_visible_left_shoulder and is_visible_right_shoulder


def check_vertical(lower_joint, upper_joint):
    threshold = 0.2
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
    
    if(check_vertical(left_hip,left_shoulder) and check_vertical(right_hip,right_shoulder)):
        return True
    
    #print(right_shoulder)
    #print(left_shoulder)

    return False



    

def classify_pose(pose_landmarks: landmark_pb2.NormalizedLandmarkList) -> str:
    # Classify the pose based on the pose landmarks
    # print("pose landmarks: {}".format(pose_landmarks))
    print(is_standing(pose_landmarks))

    return "UNKNOWN"


def print_result(detection_result: vision.PoseLandmarkerResult, output_image: mp.Image,
                 timestamp_ms: int):
    global to_window
    global last_timestamp_ms
    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms
    # print("pose landmarker result: {}".format(detection_result))
    #print(detection_result)
    if(len(detection_result.pose_landmarks) > 0):
        classify_pose(detection_result.pose_landmarks[0])
    to_window = cv2.cvtColor(
        draw_landmarks_on_image(output_image.numpy_view(), detection_result), cv2.COLOR_RGB2BGR)



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