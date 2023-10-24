import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

def euclidean_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def compute_angle(p1, p2, p3):
    # Vectors
    a = np.array([p1.x - p2.x, p1.y - p2.y])
    b = np.array([p3.x - p2.x, p3.y - p2.y])
    # Dot product and magnitudes
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Angle
    cos_theta = dot_product / (norm_a * norm_b)
    angle = np.arccos(np.clip(cos_theta, -1, 1))
    return angle

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

selected_landmark_indices = [33, 133, 159, 263, 46, 70, 4, 454, 234, 10, 338, 297, 332, 61, 291, 0, 78, 14, 317, 152, 155, 337, 299, 333]

all_landmarks = []

cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
        mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        all_landmarks = []
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Process the image for hand and face landmarks
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(image_rgb)
        face_results = face_mesh.process(image_rgb)

        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                # Add hand landmarks to the combined list
                all_landmarks.extend(hand_landmarks.landmark)

        # Face landmarks and distance matrix visualization
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:

                # Extract and draw selected landmarks
                selected_landmarks = [face_landmarks.landmark[i] for i in selected_landmark_indices]
                for landmark in selected_landmarks:
                    image = cv2.circle(image, (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])), 3,
                                       (0, 255, 0), -1)

                # Add selected face landmarks to the combined list
                all_landmarks.extend(selected_landmarks)

                # Calculate the distance matrix for face landmarks
                num_landmarks = len(all_landmarks)
                dist_matrix = np.zeros((num_landmarks, num_landmarks))
                angle_matrix = np.zeros((num_landmarks, num_landmarks))
                proximity_matrix = np.zeros((num_landmarks, num_landmarks))
                for i in range(num_landmarks):
                    for j in range(num_landmarks):
                        distance = euclidean_distance(all_landmarks[i], all_landmarks[j])
                        dist_matrix[i, j] = distance
                        # Proximity
                        proximity_matrix[i, j] = 1 if distance < 0.05 else 0
                        # Angle for every third landmark to avoid over-computation
                        if j % 3 == 0 and j + 1 < num_landmarks:
                            angle_matrix[i, j] = compute_angle(all_landmarks[i], all_landmarks[j], all_landmarks[j + 1])

                # Normalize matrices to [0, 255]
                dist_img = (255 * (dist_matrix - np.min(dist_matrix)) / (
                            np.max(dist_matrix) - np.min(dist_matrix))).astype(np.uint8)
                angle_img = (255 * (angle_matrix - np.min(angle_matrix)) / (
                            np.max(angle_matrix) - np.min(angle_matrix))).astype(np.uint8)
                proximity_img = (255 * proximity_matrix).astype(np.uint8)

                # Create RGB image
                rgb_img = cv2.merge([dist_img, angle_img, proximity_img])
                rgb_img = cv2.resize(rgb_img, (image.shape[1], image.shape[0]))  # Resize to match the camera feed

                # Display combined image
                combined_image = np.hstack((image, rgb_img))
                cv2.imshow('MediaPipe Landmarks and RGB Matrix', cv2.flip(combined_image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break

cap.release()
cv2.destroyAllWindows()
