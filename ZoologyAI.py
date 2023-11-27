import numpy as np
import cv2

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

def preProcessing(image):
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)  # Convert grayscale images to RGB format
    elif image.shape[2] == 4:
        image = image[:, :, :3]  # Remove the alpha channel if it exists
        
    image = cv2.resize(image, (244, 244), interpolation=cv2.INTER_AREA)
    imageArray = np.array(image, dtype=np.float32)
    imageArray = np.expand_dims(imageArray, axis=0)
    return imageArray
  
def create_data_generator(preprocessing_function=None):
    data_generator = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        preprocessing_function=preprocessing_function
    )
    return data_generator

train_data_generator = create_data_generator(preProcessing)
train_generator = train_data_generator.flow_from_directory(
    '/Users/hamzahhamad/Desktop/data/Train',
    target_size=(244, 244),
    batch_size=32,
    class_mode='categorical'
)

val_data_generator = ImageDataGenerator(rescale=1.0/255, preprocessing_function=preProcessing)
val_generator = val_data_generator.flow_from_directory(
    '/Users/hamzahhamad/Desktop/data/Validate',
    target_size=(244, 244),
    batch_size=32,
    class_mode='categorical'
)

test_data_generator = ImageDataGenerator(rescale=1.0/255, preprocessing_function=preProcessing)
test_generator = test_data_generator.flow_from_directory(
    '/Users/hamzahhamad/Desktop/data/Test',
    target_size=(244, 244),
    batch_size=32,
    class_mode='categorical'
)

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(244, 244, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax')) 

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
num_epochs = 32
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator
)

# Evaluate the model
score = model.evaluate(test_generator)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Save the trained model
model.save('animal_detector_model.h5')
