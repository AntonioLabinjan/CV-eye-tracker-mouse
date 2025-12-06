import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

# ---- Mediapipe face mesh setup ----
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

screen_w, screen_h = pyautogui.size()

# Iris indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye landmarks for EAR (rough approx)
LEFT_EYE = [33, 160, 158, 133, 153, 144]  # upper/lower points
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Treptaj parametri
EAR_THRESHOLD = 0.25
BLINK_COOLDOWN = 0.3  # sekunde
last_blink = 0

cap = cv2.VideoCapture(0)

def get_eye_center(landmarks, eye_indices):
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    center = points.mean(axis=0)
    return center

def compute_EAR(landmarks, eye_points):
    # 2D points
    p = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_points])
    # EAR formula
    A = np.linalg.norm(p[1]-p[5])
    B = np.linalg.norm(p[2]-p[4])
    C = np.linalg.norm(p[0]-p[3])
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    ret, frame = cap.read()
    if not ret: break
    
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        
        if len(lm) > max(max(LEFT_IRIS), max(RIGHT_IRIS)):
            # ---- Eye center for cursor ----
            left_center = get_eye_center(lm, LEFT_IRIS)
            right_center = get_eye_center(lm, RIGHT_IRIS)
            eye_center = (left_center + right_center)/2
            
            screen_x = np.clip(eye_center[0] * screen_w, 0, screen_w)
            screen_y = np.clip(eye_center[1] * screen_h, 0, screen_h)
            pyautogui.moveTo(screen_x, screen_y, duration=0.01)
            cv2.circle(frame, (int(eye_center[0]*w), int(eye_center[1]*h)), 5, (0,255,0), -1)
            
            # ---- Treptaj detection ----
            left_EAR = compute_EAR(lm, LEFT_EYE)
            right_EAR = compute_EAR(lm, RIGHT_EYE)
            avg_EAR = (left_EAR + right_EAR)/2
            
            if avg_EAR < EAR_THRESHOLD and time.time() - last_blink > BLINK_COOLDOWN:
                pyautogui.click()
                last_blink = time.time()
                cv2.putText(frame, "CLICK!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow("Eye Tracker + Blink Click", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
