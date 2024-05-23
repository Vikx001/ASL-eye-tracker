

# ASL Eye Tracker

## Overview
The ASL Eye Tracker project is designed to track and interpret American Sign Language (ASL) gestures using computer vision and machine learning techniques. The project utilizes a variety of tools and libraries to preprocess data, train models, and recognize ASL gestures in real-time. The dataset used in this project is sourced from a Kaggle ASL dataset.

## Project Structure
- `asl_dataset/`: Contains the dataset used for training and testing the models.
- `asl_knn_model.joblib`: Serialized KNN model for gesture recognition.
- `asl_model.h5`: Trained model in H5 format.
- `asl_rf_model.joblib`: Serialized Random Forest model for gesture recognition.
- `eye_tracker.py`: Main script for eye tracking and gesture recognition.
- `hand_tracking.py`: Script for hand tracking using MediaPipe.
- `hand_tracking_with_knn.py`: Script for hand tracking combined with KNN-based gesture recognition.
- `preprocess_data_and_train.py`: Script for preprocessing the dataset and training the models.
- `shape_predictor_68_face_landmarks.dat`: Pre-trained model for face landmark detection.
- `train_generator.npy`: Preprocessed training data.
- `train_knn_model.py`: Script for training the KNN model.

## Installation
To run this project, you need to have Python 3.9 installed along with the required libraries. You can install the required libraries using the following command:

```sh
pip install -r requirements.txt
```

## Usage
1. **Preprocess Data and Train Models**:
   Run the script to preprocess the dataset and train the models. The dataset is collected from a Kaggle ASL dataset.
   ```sh
   python preprocess_data_and_train.py
   ```

2. **Run Eye Tracker**:
   Execute the main script to start the eye tracking and ASL gesture recognition.
   ```sh
   python eye_tracker.py
   ```

3. **Hand Tracking with KNN**:
   Use this script for hand tracking combined with KNN-based gesture recognition.
   ```sh
   python hand_tracking_with_knn.py
   ```

## Training with Multiple Models
This project includes training data with multiple machine learning algorithms such as K-Nearest Neighbors (KNN), Random Forest, and deep learning models. Users can also train the dataset using different algorithms and compare their outputs.

- **KNN Model**:
  ```sh
  python train_knn_model.py
  ```

- **Other Models**: Modify and run the `preprocess_data_and_train.py` script to train using different algorithms. 

## Files Description
- **asl_knn_model.joblib**: This file contains the K-Nearest Neighbors (KNN) model trained on the ASL dataset.
- **asl_model.h5**: This file contains the trained ASL gesture recognition model in H5 format.
- **asl_rf_model.joblib**: This file contains the Random Forest model trained on the ASL dataset.
- **eye_tracker.py**: Main script to initialize and run the eye tracker and ASL gesture recognition.
- **hand_tracking.py**: Script to perform hand tracking using MediaPipe.
- **hand_tracking_with_knn.py**: Combines hand tracking with KNN for ASL gesture recognition.
- **preprocess_data_and_train.py**: Script to preprocess the data and train the ASL gesture recognition models.
- **shape_predictor_68_face_landmarks.dat**: Pre-trained dlib model for detecting 68 face landmarks.
- **train_generator.npy**: Numpy array containing preprocessed training data.
- **train_knn_model.py**: Script to train the KNN model on the ASL dataset.

## Requirements
- Python 3.9
- OpenCV
- MediaPipe
- PyAutoGUI
- Numpy
- Joblib
- TensorFlow
- Dlib

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

---

This README provides a comprehensive overview of your project, its structure, installation instructions, usage, and other essential details, including the additional information about the dataset and the use of multiple models. Adjust the content as needed to fit your specific project requirements.
