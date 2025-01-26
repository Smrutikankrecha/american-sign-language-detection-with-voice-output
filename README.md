# American Sign Language Dataset Preprocessing with Voice Feedback

This repository contains a Python-based project for preprocessing images of American Sign Language (ASL) gestures and extracting hand landmarks using MediaPipe. The extracted data is saved as a pickle file, which can be used for training machine learning models.

## Features
- Extracts hand landmarks from images using MediaPipe Hands module.
- Organizes data into labeled folders corresponding to ASL letters (A-Z).
- Saves preprocessed data as a pickle file for easy use in ML pipelines.
- Includes error handling and debugging logs to identify issues in the dataset.

## Prerequisites
Before running the project, ensure you have the following installed:

- Python 3.7 or higher
- OpenCV
- MediaPipe 0.10.5

You can install the required libraries with:
```bash
pip install opencv-python mediapipe
```

## Folder Structure
Organize your dataset folder (`./data`) as follows:
```
./data
    /A
        001.jpg
        002.jpg
        ...
    /B
        001.jpg
        002.jpg
        ...
    ...
```
Each subfolder corresponds to an ASL letter and contains images of hands making that gesture.

## Code Breakdown
### 1. Preprocessing Images
The `preprocess_image` function processes each image to extract hand landmarks. It returns normalized landmark points or skips the image if no hand is detected.

```python
def preprocess_image(img_path):
    """Preprocesses the image to extract hand landmarks."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image {img_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))

        return data_aux
    else:
        print(f"No hands detected in {img_path}")
        return None
```

### 2. Creating the Pickle File
The `create_pickle_file` function processes all images in the dataset folder, extracts landmarks, and saves the data and labels to a pickle file.

```python
def create_pickle_file(data_dir, pickle_filename):
    """Creates a pickle file containing processed hand landmark data."""
    data = []
    labels = []

    for dir_ in os.listdir(data_dir):
        class_path = os.path.join(data_dir, dir_)
        if not os.path.isdir(class_path):
            continue
        for img_path in os.listdir(class_path):
            img_full_path = os.path.join(class_path, img_path)
            processed_data = preprocess_image(img_full_path)
            if processed_data:
                data.append(processed_data)
                labels.append(dir_)

    if len(data) == 0:
        print("Error: No data collected. Check your images and folder structure.")
        return

    with open(pickle_filename, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Pickle file {pickle_filename} created successfully.")
```

### 3. Main Script
The main script calls the `create_pickle_file` function with the dataset directory and output pickle file name.

```python
if __name__ == "__main__":
    DATA_DIR = './data'
    pickle_filename = 'data_1.pickle'
    create_pickle_file(DATA_DIR, pickle_filename)
```

## Running the Code
1. Organize your dataset as described in the folder structure section.
2. Run the script:
   ```bash
   python script_name.py
   ```
3. The processed data will be saved as `data_1.pickle` in the current directory.

## Debugging Tips
- If the script prints "No hands detected" for multiple images, ensure that the images have clear and visible hands.
- Check for missing or unreadable images in the dataset.
- Ensure the folder structure matches the expected format.

## Contributing
If you'd like to improve this project, feel free to fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

