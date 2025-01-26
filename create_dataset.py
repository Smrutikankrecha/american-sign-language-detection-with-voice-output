import os
import pickle
import cv2
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)

# Define data directory
DATA_DIR = './data'


def preprocess_image(img_path):
    """Preprocesses the image to extract hand landmarks."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image {img_path}")
        return None  # Skip invalid image

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Check if any hand landmarks are detected
    if results.multi_hand_landmarks:
        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            # Extract normalized landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize landmarks and add to data_aux
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))

        return data_aux
    else:
        print(f"No hands detected in {img_path}")
        return None


def create_pickle_file(data_dir, pickle_filename):
    """Creates a pickle file containing processed hand landmark data."""
    data = []
    labels = []

    # Traverse through each class (folder) and process images
    for dir_ in os.listdir(data_dir):
        class_path = os.path.join(data_dir, dir_)
        if not os.path.isdir(class_path):
            continue  # Skip if it's not a directory
        for img_path in os.listdir(class_path):
            img_full_path = os.path.join(class_path, img_path)

            # Preprocess the image and get the landmarks
            processed_data = preprocess_image(img_full_path)
            if processed_data:
                data.append(processed_data)
                labels.append(dir_)

    # Ensure that data was collected
    if len(data) == 0:
        print("Error: No data collected. Check your images and folder structure.")
        return

    # Write the data and labels to a pickle file
    with open(pickle_filename, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Pickle file {pickle_filename} created successfully.")


if __name__ == "__main__":
    pickle_filename = 'data.pickle'
    create_pickle_file(DATA_DIR, pickle_filename)
