import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
# Define a global variable to hold the loaded model
loaded_model = None

def load_model(model_path):
    """Load the model."""
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return loaded_model
    except FileNotFoundError:
        print(f"Model file not found: {model_path}. Continuing without loading the model.")
    
    

def load_synset(synset_path):
    """Load class names from the synset file."""
    class_names = []
    with open(synset_path, 'r') as f:
        for line in f:
            class_names.append(line.strip())
            print(class_names)
    return class_names

    
def predict_from_file(model_path, synset_path, img_path):
    try:
        loaded_model = load_model(model_path)
        print("Model Used: " + model_path)
        # Load class names
        class_names = load_synset(synset_path)
        img = tf.keras.utils.load_img(img_path, target_size=(224,224))#download your own image)

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Add the image to a batch where it's the only member.

        # Preprocess the image
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

        predictions = loaded_model.predict(img_array)[0]
        top_indices = np.argsort(predictions)[::-1][:5]  # Get indices of top 5 predictions
        top_classes = [class_names[i] for i in top_indices]
        top_probabilities = [predictions[i] * 100 for i in top_indices]  # Convert to percentage

        response = {'top_classes': top_classes, 'probabilities': top_probabilities}
        return response
    except Exception as e:
        return str(e)



