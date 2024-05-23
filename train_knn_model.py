# train_knn_model.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# Set the paths
dataset_dir = "asl_dataset"  # Update this with the actual path to the dataset
asl_classes = sorted(os.listdir(dataset_dir))

# Parameters
img_height, img_width = 64, 64
batch_size = 32

# Data generators
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Function to extract features and labels from the generator
def extract_features_and_labels(generator):
    features, labels = [], []
    for _ in range(len(generator)):
        x, y = next(generator)  # Use next(generator) to get the next batch
        features.extend(x)
        labels.extend(y)
    return np.array(features), np.array(labels)

# Extract features and labels from the generators
X_train, y_train = extract_features_and_labels(train_generator)
X_val, y_val = extract_features_and_labels(validation_generator)

# Flatten the images for sklearn models
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train, y_train.argmax(axis=1))

# Evaluate the model
y_pred = knn_model.predict(X_val)
accuracy = accuracy_score(y_val.argmax(axis=1), y_pred)
print(f"KNN Accuracy: {accuracy}")

# Save the model
dump(knn_model, 'asl_knn_model.joblib')
