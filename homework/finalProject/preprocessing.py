# Import libraries
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv

model_path = '/Users/y/Desktop/344MLhomework/homework/finalProject/model/hand_landmarker.task'
input_folder = '/Users/y/Desktop/344MLhomework/homework/finalProject/input_images'
output_csv = '/Users/y/Desktop/344MLhomework/homework/finalProject/landmarks_data.csv'

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