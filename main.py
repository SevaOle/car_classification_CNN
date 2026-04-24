from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

BASE_IMAGE_DIR = (
    Path.cwd()
    / "data"
    / "car-make-model-and-generation"
    / "car-dataset-200"
    / "riotu-cars-dataset-200"
)

CSV_PATH = Path("split.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 30

target_column = "make_id"
class_name_column = "make"


# Loading the dataset
df = pd.read_csv(CSV_PATH)

print(df.head())
print("\nDataset shape:", df.shape)

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()
test_df = df[df["split"] == "test"].copy()

print()
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

print()
num_classes = df[target_column].nunique()
print("Number of classes:", num_classes)
class_names = df[[target_column, class_name_column]].drop_duplicates()
class_names = class_names.sort_values(target_column)[class_name_column]
class_names = class_names.tolist()
print(class_names)


# Set up full paths for training images
def get_paths_and_labels(dataframe):
    paths = []
    labels = []

    for column, row in dataframe.iterrows():
        full_path = BASE_IMAGE_DIR / row["filepath"]

        paths.append(str(full_path))
        labels.append(row[target_column])

    return paths, labels

print("Getting full paths and labels for training, validation, and test sets...")
train_paths, train_labels = get_paths_and_labels(train_df)
val_paths, val_labels = get_paths_and_labels(val_df)
test_paths, test_labels = get_paths_and_labels(test_df)

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))


print("\nLoading images and preprocessing...")

def load_and_preprocess_image(filepath, label):
    image = tf.io.read_file(filepath)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)

    # Normalizing pixel values from [0, 255] to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


train_dataset = train_dataset.map(load_and_preprocess_image)
val_dataset = val_dataset.map(load_and_preprocess_image)
test_dataset = test_dataset.map(load_and_preprocess_image)

def find_bad_files(train_paths, val_paths, test_paths):
    bad_files = []

    for path in train_paths + val_paths + test_paths:
        try:
            image_data = tf.io.read_file(path)
            image = tf.image.decode_image(
                image_data,
                channels=3,
                expand_animations=False
            )
        except Exception:
            bad_files.append(path)

    print("Bad files found:", len(bad_files))
    with open("bad_files.txt", "w") as f:
        for bad_file in bad_files:
            f.write(bad_file + "\n")

def delete_bad_files():
    with open("bad_files.txt", "r") as f:
        bad_files = f.read().splitlines()

    for bad_file in bad_files:
        try:
            os.remove(bad_file)
            print(f"Deleted bad file: {bad_file}")
        except Exception as e:
            print(f"Error deleting {bad_file}: {e}")

# find_bad_files(train_paths, val_paths, test_paths)
# delete_bad_files()
# exit()
# had to also rerun initialize_data after that to update the csv with the bad files removed



print("\nBatching the datasets...")
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

print("\nDisplaying sample training images...")

plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[int(labels[i])])
        plt.axis("off")

plt.tight_layout()
plt.show()

print("\nBuilding CNN model...")

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.05),
    layers.RandomRotation(0.02),
])

model = keras.Sequential([
    keras.Input(shape=IMG_SIZE + (3,)),

    data_augmentation,

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation="softmax")
])

model.summary()


print("\nCompiling...")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nTraining the model...")
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

#history = model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=EPOCHS,
#     callbacks=[early_stopping]
# )
#model.save("car_make_cnn.keras")

print("\nTesting...")

test_loss, test_accuracy = model.evaluate(test_dataset)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history.csv", index=False)

####### Results #######

# print("\nPlotting training history...")
#
# plt.figure(figsize=(8, 5))
# plt.plot(history["accuracy"], label="Training Accuracy")
# plt.plot(history["val_accuracy"], label="Validation Accuracy")
# plt.title("Training and Validation Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(8, 5))
# plt.plot(history["loss"], label="Training Loss")
# plt.plot(history["val_loss"], label="Validation Loss")
# plt.title("Training and Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()