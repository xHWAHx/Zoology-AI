import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

def preProcessing(image):
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    image = cv2.resize(image, (244, 244), interpolation=cv2.INTER_AREA)
    imageArray = np.array(image, dtype=np.float32)
    imageArray = imageArray / 255.0  # Normalize the image
    imageArray = np.expand_dims(imageArray, axis=0)
    return imageArray

def pre_process_image(image_path):
    image = cv2.imread(image_path)
    preprocessed_image = preProcessing(image)
    return preprocessed_image

# Predict the class of the input image
def predict_animal(image_path, model):
    preprocessed_image = pre_process_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions, axis=1)
    return predicted_class_index

# Load the saved model
model_path = 'animal_detector_model.h5'
loaded_model = load_model(model_path)

# Test on a new image
image_path = '/Users/hamzahhamad/Desktop/test/butterflytest.jpeg'  # Replace with your actual image path
predicted_class_index = predict_animal(image_path, loaded_model)

# Define the class labels
class_labels = ['Butterfly', 'Cow', 'Dog', 'Elephant', 'Hen', 'Horse', 'Sheep', 'Squirrel']  # Replace with your actual class names

# Map the predicted class index to the class label
class_label = class_labels[predicted_class_index[0]]
print(f'The predicted animal class is: {class_label}')

# Show the preprocessed image
preprocessed_image = pre_process_image(image_path)
plt.imshow(preprocessed_image[0])
plt.title("Preprocessed Image")
plt.show()
