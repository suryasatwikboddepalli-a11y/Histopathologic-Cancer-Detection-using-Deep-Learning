# train_model.py (Improved Version)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# We import a new tool called a "Callback"
from tensorflow.keras.callbacks import ReduceLROnPlateau

print("TensorFlow version:", tf.__version__)

# --- Configuration ---
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
DATA_DIR = 'D:/Projects/Datasets/organized_train_data'

# --- Load and Prepare Data ---
print(f"Loading data from: {DATA_DIR}")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20, # Added more augmentation
    zoom_range=0.2
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# --- Build the Model ---
print("Building the model...")
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# --- Compile the Model ---
print("Compiling the model...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- ADDED: Learning Rate Scheduler ---
# This will reduce the learning rate when validation accuracy plateaus
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', 
    patience=2, 
    verbose=1, 
    factor=0.5, 
    min_lr=0.00001
)

# --- Train the Model ---
# We increased epochs from 3 to 15 for much better training.
print("\nStarting improved model training... This will take significantly longer!")
history = model.fit(
    train_generator,
    epochs=15, 
    validation_data=validation_generator,
    callbacks=[learning_rate_reduction] # We add our new tool here
)

# --- Save the Final Model ---
print("Training complete! Saving the improved model...")
model.save('histopathology_model.h5')

print("\nNew, improved model saved as 'histopathology_model.h5'. Please restart your web app!")