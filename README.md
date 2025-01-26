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

