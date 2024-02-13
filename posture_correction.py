import cv2
import math
import serial
import time
import mediapipe as mp
import numpy as np
import sys

def find_distance(a, b, w, h):
    a = np.array(a)
    b = np.array(b)
    a[0] = a[0] * w
    a[1] = a[1] * h
    b[0] = b[0] * w
    b[1] = b[1] * h
    dist = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    return dist

def compute_sit_angles(a, b, w, h):
    a = np.array(a)
    b = np.array(b)
    a[0] = a[0] * w
    a[1] = a[1] * h
    b[0] = b[0] * w
    b[1] = b[1] * h
    theta = math.acos((b[1] - a[1]) * (-a[1]) / (math.sqrt((b[0] - b[1]) ** 2 + (b[1] - a[1]) ** 2) * a[1]))
    degree = int(180/math.pi) * theta
    return degree

def compute_exercise_angles(a, b, c):
    x1, y1, z1 = a
    x2, y2, z2 = b
    x3, y3, z3 = c
    # Calculate vectors CA and CB
    CA = np.array([x1-x3, y1-y3, z1-z3])
    CB = np.array([x2-x3, y2-y3, z2-z3])
    
    # Calculate dot product of CA and CB
    dot_product = np.dot(CA, CB)
    
    # Calculate magnitudes of vectors CA and CB
    magnitude_CA = np.linalg.norm(CA)
    magnitude_CB = np.linalg.norm(CB)
    
    # Calculate angle theta using the dot product
    theta = np.arccos(dot_product / (magnitude_CA * magnitude_CB))
    
    # Convert angle from radians to degrees
    theta_degrees = np.degrees(theta)
    
    return theta_degrees

def send_message(port, command):
    if command =="G":
        port.write(b'G') 
    elif command =="R":
        port.write(b'R')
    elif command =="V":
        port.write(b'V')
    elif command =="O":
        port.write(b'O') 
    return

def extract_top_landmarks(landmarks, direction, mp_pose):
    if direction == "left":
        ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    if direction == "right":
        ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    return ear, shoulder, hip

def extract_squat_landmarks(landmarks, direction, mp_pose):
    if direction == "left":
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
    if direction == "right":
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]
    return knee, hip, shoulder, foot

def extract_push_up_landmarks(landmarks, direction, mp_pose):
    if direction == "left":
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
    if direction == "right":
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
    return wrist, elbow, shoulder, ankle, hip

def process_frame(frame, status):
    if status == "pre":
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
    if status == "post":
        frame.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return image

def compute_time(frame, fps, neck_angle, torso_angle, good_frames, bad_frames, font, colors):
    angle_text_string = 'Neck : ' + str(int(neck_angle)) + '  Torso : ' + str(int(torso_angle))
    if neck_angle < 40 and torso_angle < 10:
        bad_frames = 0
        good_frames += 1
        cv2.putText(frame, angle_text_string, (10, 30), font, 0.9, colors["light_green"], 2)
    else:
        good_frames = 0
        bad_frames += 1
        cv2.putText(frame, angle_text_string, (10, 30), font, 0.9, colors["red"], 2)
    good_time = (1/fps) * good_frames
    bad_time = (1/fps) * bad_frames
    return frame, good_frames, bad_frames, good_time, bad_time

def compute_squat(l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, l_foot, r_foot):
    # condition 1
    l_hip_angle = compute_exercise_angles(l_knee, l_hip, l_shoulder)
    r_hip_angle = compute_exercise_angles(r_knee, r_hip, r_shoulder)
    hip_angle = (l_hip_angle + r_hip_angle) / 2

    # condition 2
    def compute_knee_hip_height(knee, hip):
        _, knee_y, _ = knee
        _, hip_y, _ = hip
        return round(abs(knee_y - hip_y), 3)
    l_knee_hip_height = compute_knee_hip_height(l_knee, l_hip)
    r_knee_hip_height = compute_knee_hip_height(r_knee, r_hip)

    # condition 3
    def compute_foot_knee_width(foot, knee):
        foot_x, _, _ = foot
        knee_x, _, _ = knee
        return round(abs(foot_x - knee_x), 3)
    l_foot_knee_width = compute_foot_knee_width(l_foot, l_knee)
    r_foot_knee_width = compute_foot_knee_width(r_foot, r_knee)

    return hip_angle, l_knee_hip_height, r_knee_hip_height, l_foot_knee_width, r_foot_knee_width

def isCorrectSquat(hip_angle, l_knee_hip_height, r_knee_hip_height, l_foot_knee_width, r_foot_knee_width):
    squat_guide = ""
    is_squat_performed = True
    are_conditions_met = (60 <= hip_angle <= 120 and l_knee_hip_height <= 0.2 and r_knee_hip_height <= 0.2 and l_foot_knee_width <= 0.1 and r_foot_knee_width <= 0.1)

    if are_conditions_met:
        is_squat_performed = False
    else:
        if is_squat_performed == False:
            is_squat_performed = True
        # build guide given the wrong conditions
        if hip_angle < 60:
            squat_guide += "Increase the height of the hip\n"
        if hip_angle > 120:
            squat_guide += "Decrease the height of the hip\n"
        if l_knee_hip_height > 0.2 or r_knee_hip_height > 0.2:
            squat_guide += "Keep the thigh horizontal to the floor\n"
        if l_foot_knee_width > 0.1 or r_foot_knee_width > 0.1:
            squat_guide += "Do not exceed the tip of the toe with your knee\n"
    
    return is_squat_performed, squat_guide

def compute_push_up(l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_ankle, r_ankle):
    # Condition 1
    l_elbow_angle = compute_exercise_angles(l_wrist, l_elbow, l_shoulder)
    r_elbow_angle = compute_exercise_angles(r_wrist, r_elbow, r_shoulder)
    elbow_angle = (l_elbow_angle + r_elbow_angle) / 2

    # Condition 2
    l_body_angle = compute_exercise_angles(l_ankle, l_hip, l_shoulder)
    r_body_angle = compute_exercise_angles(r_ankle, r_hip, r_shoulder)
    body_angle = (l_body_angle + r_body_angle) / 2
    return elbow_angle, body_angle

def isCorrectPushUp(elbow_angle, body_angle):
    push_up_guide = ""
    is_push_up_performed = True

    if 70 <= elbow_angle <= 100 and 160 <= body_angle <= 200:
        is_push_up_performed = False
    else:
        if is_push_up_performed == False:
            is_push_up_performed = True
        if elbow_angle < 70:
            push_up_guide += "Increase the height of the elbow\n"
        if elbow_angle > 100:
            push_up_guide += "Decrease the height of the elbow\n"
        if body_angle < 160:
            push_up_guide += "Increase the height of the body\n"
        if body_angle > 200:
            push_up_guide += "Decrease the height of the body\n"
    return is_push_up_performed, push_up_guide

def posture_correction(posture, mp_drawing, mp_pose, font, colors):
    cap = cv2.VideoCapture(0)
    '''
    arduino = serial.Serial('/dev/ttyACM2', 9600)
    time.sleep(2)
    '''
    if posture == "sit":
        good_frames = 0
        bad_frames = 0
        good_time = 0
        bad_time = 0
    elif posture == "squat" or posture == "pushup":
        repetitions = 0
    else:
        print("Invalid posture")
        return
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            h, w = frame.shape[:2]
            frame = process_frame(frame, "pre")
            results = pose.process(frame)
            frame = process_frame(frame, "post")
            try:
                landmarks = results.pose_landmarks.landmark
                if posture == "sit":
                    l_ear, l_shoulder, l_hip = extract_top_landmarks(landmarks, "left", mp_pose)
                    r_ear, r_shoulder, r_hip = extract_top_landmarks(landmarks, "right", mp_pose)
                    offset = find_distance(l_shoulder, r_shoulder, w, h)
                    if offset < 100:
                        cv2.putText(frame, str(int(offset)) + " Aligned", (w-150, 30), font, 0.9, colors["green"], 2)
                        neck_angle = compute_sit_angles(r_shoulder, r_ear, w, h)
                        torso_angle = compute_sit_angles(r_hip, r_shoulder, w, h)

                        frame, good_frames, bad_frames, good_time, bad_time = compute_time(frame, fps, neck_angle, torso_angle, good_frames, bad_frames, font, colors)                

                        if good_time > 0:
                            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                            cv2.putText(frame, time_string_good, (10, h - 20), font, 0.9, colors["green"], 2)
                        else:
                            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                            cv2.putText(frame, time_string_bad, (10, h - 20), font, 0.9, colors["red"], 2)

                        if good_time > 1:
                            #send_message(arduino, "G")
                            pass
                        if bad_time > 1:
                            #send_message(arduino, "R")
                            pass
                    else:
                        cv2.putText(frame, str(int(offset)) + " Not Aligned", (w-150, 30), font, 0.9, colors["red"], 2)
                        #send_message(arduino, "V")
                else:
                    if posture == "squat":
                        l_knee, l_hip, l_shoulder, l_foot = extract_squat_landmarks(landmarks, "left", mp_pose)
                        r_knee, r_hip, r_shoulder, r_foot = extract_squat_landmarks(landmarks, "right", mp_pose)
                        hip_angle, l_knee_hip_height, r_knee_hip_height, l_foot_knee_width, r_foot_knee_width = compute_squat(l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, l_foot, r_foot)
                        is_exercise_performed, exercise_guide = isCorrectSquat(hip_angle, l_knee_hip_height, r_knee_hip_height, l_foot_knee_width, r_foot_knee_width)
                    if posture == "pushup":
                        l_wrist, l_elbow, l_shoulder, l_ankle, l_hip = extract_push_up_landmarks(landmarks, "left", mp_pose)
                        r_wrist, r_elbow, r_shoulder, r_ankle, r_hip = extract_push_up_landmarks(landmarks, "right", mp_pose)
                        elbow_angle, body_angle = compute_push_up(l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_ankle, r_ankle)
                        is_exercise_performed, exercise_guide = isCorrectPushUp(elbow_angle, body_angle)
                    if not is_exercise_performed:
                        repetitions += 1

                    cv2.rectangle(frame, (0,0), (400,73), (245,117,16), -1)
                    # Rep data
                    cv2.putText(frame, 'REPETITIONS: '+str(repetitions), (15,12), 
                                font, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(frame, "GUIDE: ", 
                                (15,36), 
                                font, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    if exercise_guide != "":
                        for i, guide in enumerate(exercise_guide.split('\n')):
                            t = i+1
                            cv2.putText(frame, guide, 
                                (15,36 + t*12), 
                                font, 0.5, (255,255,255), 1, cv2.LINE_AA)
            except:
                pass
            # Render detections
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            cv2.imshow('Mediapipe Feed', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                #send_message(arduino, "O")
                break
        cap.release()
        cv2.destroyAllWindows()
        #arduino.close()
    return


def main():
    args = sys.argv[1:]
    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(args) == 2 and args[0] == "-posture":
        posture = args[1]

        colors = {
            "blue": (255, 127, 0),
            "red": (50, 50, 255),
            "green": (127, 255, 0),
            "dark_blue": (127, 20, 0),
            "light_green": (127, 233, 100),
            "yellow": (0, 255, 255),
            "pink": (255, 0, 255),
            "black": (0, 0, 0)
        }

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        posture_correction(posture, mp_drawing, mp_pose, font, colors)
    else:
        print("Invalid arguments")
    return

if __name__ == "__main__":
    main()