# Import libraries
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv
import numpy as np
import cv2

model_path = '/Users/y/Desktop/344MLhomework/homework/finalProject/model/hand_landmarker.task'
input_folder = '/Users/y/Desktop/hands/right'
output_csv = '/Users/y/Desktop/344MLhomework/homework/finalProject/landmarks_data_right.csv'

# 2. Prepare the CSV header
# Columns: Filename, Handedness, Score, then x, y, z for all 21 landmarks
header = ['filename', 'hand_label', 'score']
for i in range(21):
    header.extend([f'x_{i}', f'y_{i}', f'z_{i}'])


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

'''
# Visualization
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

'''
with HandLandmarker.create_from_options(options) as landmarker:
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Loop through images
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(input_folder, filename)
                
                # Load image as MediaPipe Image
                try:
                    mp_image = mp.Image.create_from_file(file_path)
                    
                except Exception as e:
                    print(f"Skipping {filename}: {e}")
                    continue

                # Detect hands
                detection_result = landmarker.detect(mp_image)
                
                '''
                img = mp_image.numpy_view()

                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                
                annotated_image = draw_landmarks_on_image(img, detection_result)
                cv2.imshow("Annotated Image", annotated_image)
                cv2.waitKey(0)
                '''
                # The result contains lists of hands. We assume they match by index.
                # detection_result.hand_landmarks is a list of lists (one list per hand)
                # detection_result.handedness is a list of lists (one category per hand)
                
                if detection_result.hand_landmarks:
                    # Zip pairs the landmarks with the correct handedness info
                    for landmarks, handedness_list in zip(detection_result.hand_landmarks, detection_result.handedness):
                        
                        # Extract Label (Left/Right) and Score
                        hand_info = handedness_list[0] # Get the first category
                        label = hand_info.category_name
                        score = hand_info.score
                        
                        # Start the row data
                        row = [filename, label, score]
                        
                        # Extract the 21 landmarks (Normalized)
                        for landmark in landmarks:
                            row.extend([landmark.x, landmark.y, landmark.z])
                            
                        writer.writerow(row)
                    print(f"Saved: {filename}")
                else:
                    print(f"No hands: {filename}")

print(f"\nDone! Results saved to {output_csv}")


