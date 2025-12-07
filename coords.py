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

# ---- Iris indices ----
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ---- Eye landmarks for EAR ----
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# ---- Blink parameters ----
EAR_THRESHOLD = 0.25
BLINK_COOLDOWN = 0.3
last_blink = 0

cap = cv2.VideoCapture(0)

def get_eye_center(landmarks, eye_indices):
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    return pts.mean(axis=0)

def compute_EAR(landmarks, eye_points):
    p = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_points])
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # ---- Eye centers ----
        left_center = get_eye_center(lm, LEFT_IRIS)
        right_center = get_eye_center(lm, RIGHT_IRIS)
        eye_center = (left_center + right_center) / 2

        # ---- Mapping to screen ----
        norm_x, norm_y = eye_center
        screen_x = np.clip(norm_x * screen_w, 0, screen_w)
        screen_y = np.clip(norm_y * screen_h, 0, screen_h)

        pyautogui.moveTo(screen_x, screen_y, duration=0.01)

        # ---- Draw center ----
        cx, cy = int(norm_x * w), int(norm_y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # ---- Coordinate overlays ----
        cv2.putText(
            frame,
            f"Eye norm: x={norm_x:.3f}, y={norm_y:.3f}",
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Screen: x={int(screen_x)}, y={int(screen_y)}",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"L eye: ({left_center[0]:.3f}, {left_center[1]:.3f})",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )

        cv2.putText(
            frame,
            f"R eye: ({right_center[0]:.3f}, {right_center[1]:.3f})",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )

        # ---- Blink detection ----
        left_EAR = compute_EAR(lm, LEFT_EYE)
        right_EAR = compute_EAR(lm, RIGHT_EYE)
        avg_EAR = (left_EAR + right_EAR) / 2

        if avg_EAR < EAR_THRESHOLD and time.time() - last_blink > BLINK_COOLDOWN:
            pyautogui.click()
            last_blink = time.time()
            cv2.putText(
                frame,
                "CLICK!",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    cv2.imshow("Eye Tracker + Blink Click + Coords", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
