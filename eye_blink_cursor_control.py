import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# EAR calculation function
def calculate_ear(landmarks, eye_indices):
    eye = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define EAR threshold and consecutive frame count
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3

# Indices for left and right eye landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

blink_counter = 0
screen_width, screen_height = pyautogui.size()

# Function to map coordinates from the frame to the screen with sensitivity adjustment
def map_to_screen(x, y, frame_width, frame_height, sensitivity=2.0):
    # Adjust the mapping to cover the screen more effectively
    screen_x = np.interp(x, [0, frame_width], [0, screen_width * sensitivity]) - screen_width * (sensitivity - 1) / 2
    screen_y = np.interp(y, [0, frame_height], [0, screen_height * sensitivity]) - screen_height * (sensitivity - 1) / 2
    screen_x = np.clip(screen_x, 0, screen_width)
    screen_y = np.clip(screen_y, 0, screen_height)
    return screen_x, screen_y

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_height, frame_width, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                )

                landmarks = face_landmarks.landmark
                left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
                right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
                ear = (left_ear + right_ear) / 2.0

                if ear < EYE_AR_THRESH:
                    blink_counter += 1
                else:
                    if blink_counter >= EYE_AR_CONSEC_FRAMES:
                        pyautogui.click()
                    blink_counter = 0

                # Get the nose landmark for cursor movement
                nose = landmarks[1]
                nose_x, nose_y = int(nose.x * frame_width), int(nose.y * frame_height)

                # Invert the movement direction for natural control
                nose_x = frame_width - nose_x
                nose_y = frame_height - nose_y

                # Convert nose position to screen position with sensitivity adjustment
                screen_x, screen_y = map_to_screen(nose_x, nose_y, frame_width, frame_height, sensitivity=3.0)

                # Move the cursor
                pyautogui.moveTo(screen_x, screen_y, duration=0.1)

        cv2.imshow('Eye Blink Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
