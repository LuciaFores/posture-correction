import cv2
import math
import serial
import serial.tools.list_ports as port_list
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
    elif command =="Y":
        port.write(b'Y')
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

def compute_time(fps, neck_angle, torso_angle, good_frames, bad_frames):
    sit_guide = ""
    if neck_angle < 70 and torso_angle < 10:
        bad_frames = 0
        good_frames += 1
    else:
        good_frames = 0
        bad_frames += 1
    
    if neck_angle > 70:
        sit_guide += "Increase the height of the neck\n"
    if torso_angle > 10:
        sit_guide += "Increase the height of the torso\n"
    
    good_time = (1/fps) * good_frames
    bad_time = (1/fps) * bad_frames
    return good_frames, bad_frames, good_time, bad_time, sit_guide

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
    is_squat_correct = False
    are_conditions_met = (60 <= hip_angle <= 120 and l_knee_hip_height <= 0.2 and r_knee_hip_height <= 0.2 and l_foot_knee_width <= 0.1 and r_foot_knee_width <= 0.1)

    if are_conditions_met:
        is_squat_correct = True
    elif hip_angle < 40:
        is_squat_correct = None
    else:
        # build guide given the wrong conditions
        if 40 <= hip_angle < 60:
            squat_guide += "Increase the height of the hip\n"
        if hip_angle > 120:
            squat_guide += "Decrease the height of the hip\n"
        if l_knee_hip_height > 0.2 or r_knee_hip_height > 0.2:
            squat_guide += "Keep the thigh horizontal to the floor\n"
        if l_foot_knee_width > 0.1 or r_foot_knee_width > 0.1:
            squat_guide += "Do not exceed the tip of the toe with your knee\n"
    
    return is_squat_correct, squat_guide

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
    is_push_up_correct = False

    if 70 <= elbow_angle <= 100 and 160 <= body_angle <= 200:
        is_push_up_correct = True
    elif body_angle < 130:
        is_push_up_correct = None
    else:
        if elbow_angle < 70:
            push_up_guide += "Increase the height of the elbow\n"
        if elbow_angle > 100:
            push_up_guide += "Decrease the height of the elbow\n"
        if 130 <= body_angle < 160:
            push_up_guide += "Increase the height of the body\n"
        if body_angle > 200:
            push_up_guide += "Decrease the height of the body\n"
    return is_push_up_correct, push_up_guide

def posture_correction(posture, mp_drawing, mp_pose, font, colors, port):
    cap = cv2.VideoCapture(0)
    
    arduino = serial.Serial(port, 9600)
    time.sleep(2)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    video_writer = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))
    guide = ""
    if posture == "sit":
        good_frames = 0
        bad_frames = 0
        good_time = 0
        bad_time = 0
    elif posture == "squat" or posture == "pushup":
        pass
    else:
        print("Invalid posture")
        return
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            _, frame = cap.read()
            frame = process_frame(frame, "pre")
            results = pose.process(frame)
            frame = process_frame(frame, "post")
            try:
                landmarks = results.pose_landmarks.landmark
                if posture == "sit":
                    l_ear, l_shoulder, l_hip = extract_top_landmarks(landmarks, "left", mp_pose)
                    r_ear, r_shoulder, r_hip = extract_top_landmarks(landmarks, "right", mp_pose)
                    offset = find_distance(l_shoulder, r_shoulder, w, h)
                    cv2.rectangle(frame, (w-150,0), (w,40), colors["black"], -1)
                    if offset < 100:
                        cv2.putText(frame, "Aligned", (w-145, 25), font, 0.7, colors["green"], 1, cv2.LINE_AA)
                        neck_angle = compute_sit_angles(r_shoulder, r_ear, w, h)
                        torso_angle = compute_sit_angles(r_hip, r_shoulder, w, h)
                        # put on the frame the neck angle at the height of the right shoulder
                        cv2.putText(frame, "Neck Angle: " + str(round(neck_angle, 2)),
                                (int(r_shoulder[0] * w), int(r_shoulder[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        # put on the frame the torso angle at the height of the right hip
                        cv2.putText(frame, "Torso Angle: " + str(round(torso_angle, 2)),
                                (int(r_hip[0] * w), int(r_hip[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)

                        good_frames, bad_frames, good_time, bad_time, guide = compute_time(fps, neck_angle, torso_angle, good_frames, bad_frames)                

                        cv2.rectangle(frame, (0,0), (400,73), colors["light_cyan"], -1)
                        if good_time > 0:
                            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                            cv2.putText(frame, time_string_good, (15,12), 
                                font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        else:
                            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                            cv2.putText(frame, time_string_bad, (15,12), 
                                font, 0.5, colors["black"], 1, cv2.LINE_AA)

                        if good_time > 10:
                            send_message(arduino, "G")
                        elif bad_time > 10:
                            send_message(arduino, "R")
                        else:
                            send_message(arduino, "V")
                        
                    else:
                        cv2.putText(frame, "Not Aligned", (w-145, 25), font, 0.7, colors["red"], 1, cv2.LINE_AA)
                        send_message(arduino, "Y")
                else:
                    if posture == "squat":
                        l_knee, l_hip, l_shoulder, l_foot = extract_squat_landmarks(landmarks, "left", mp_pose)
                        r_knee, r_hip, r_shoulder, r_foot = extract_squat_landmarks(landmarks, "right", mp_pose)
                        hip_angle, l_knee_hip_height, r_knee_hip_height, l_foot_knee_width, r_foot_knee_width = compute_squat(l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, l_foot, r_foot)
                        # put on the frame the hip angle at the height of the left hip
                        cv2.putText(frame, "Hip Angle: " + str(round(hip_angle, 2)),
                                (int(l_hip[0] * w), int(l_hip[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        # put on the frame the knee-hip height at the height of the left knee
                        cv2.putText(frame, "Knee-Hip Height: " + str(round(l_knee_hip_height, 2)),
                                (int(l_knee[0] * w), int(l_knee[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        # put on the frame the foot-knee width at the height of the left foot
                        cv2.putText(frame, "Foot-Knee Width: " + str(round(l_foot_knee_width, 2)),
                                (int(l_foot[0] * w), int(l_foot[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        is_exercise_performed, guide = isCorrectSquat(hip_angle, l_knee_hip_height, r_knee_hip_height, l_foot_knee_width, r_foot_knee_width)
                    if posture == "pushup":
                        l_wrist, l_elbow, l_shoulder, l_ankle, l_hip = extract_push_up_landmarks(landmarks, "left", mp_pose)
                        r_wrist, r_elbow, r_shoulder, r_ankle, r_hip = extract_push_up_landmarks(landmarks, "right", mp_pose)
                        elbow_angle, body_angle = compute_push_up(l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_ankle, r_ankle)
                        # put on the frame the elbow angle at the height of the left elbow
                        cv2.putText(frame, "Elbow Angle: " + str(round(elbow_angle, 2)),
                                (int(l_elbow[0] * w), int(l_elbow[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        # put on the frame the body angle at the height of the left ankle
                        cv2.putText(frame, "Body Angle: " + str(round(body_angle, 2)),
                                (int(l_ankle[0] * w), int(l_ankle[1] * h)), font, 0.5, colors["black"], 1, cv2.LINE_AA)
                        is_exercise_performed, guide = isCorrectPushUp(elbow_angle, body_angle)

                    cv2.rectangle(frame, (0,0), (400,73), colors["light_cyan"], -1)
            except:
                pass

            if guide != "":
                cv2.putText(frame, "GUIDE: ", 
                                (15,12), 
                                font, 0.5, colors["black"], 1, cv2.LINE_AA)
                for i, guide_line in enumerate(guide.split('\n')):
                    t = i+1
                    cv2.putText(frame, guide_line, 
                        (15,12 + t*12), 
                        font, 0.5, colors["black"], 1, cv2.LINE_AA)
            # Render detections
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=colors["pink"], thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=colors["blue"], thickness=2, circle_radius=2) 
                                    )               
            cv2.imshow('Mediapipe Feed', frame)
            video_writer.write(frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                send_message(arduino, "O")
                break
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        arduino.close()
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
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "light_cyan" : (245,117,16)
        }

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        port = list(port_list.comports())
        port = str(port[0].device)

        posture_correction(posture, mp_drawing, mp_pose, font, colors, port)
    else:
        print("Invalid arguments")
    return

if __name__ == "__main__":
    main()