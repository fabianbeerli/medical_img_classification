import numpy as np
import tensorflow as tf

def load_image(img_path, target_size=(224, 224)):
    """Load and preprocess the input image."""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

def predict_category(model_path, img_path, class_names, top_k=5):
    """Load the model and predict the top k classes of the input image."""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the input image
    img_array = load_image(img_path)
    
    # Perform prediction
    predictions = model.predict(img_array)[0]
    
    # Get the top k predicted class indices and probabilities
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_probabilities = predictions[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    # Convert probabilities to percentages
    top_probabilities_percentage = (top_probabilities * 100).tolist()
    
    return top_classes, top_probabilities_percentage


def load_synset(synset_path):
    class_names = []
    with open(synset_path, 'r') as f:
        for line in f:
            class_names.append(line.strip())
    return class_names

def predict_from_file(model_path, synset_path, img_path):
    try:
        # Load class names
        class_names = load_synset(synset_path)
        
        # Predict the top classes of the input image
        top_classes, top_probabilities = predict_category(model_path, img_path, class_names)
        
        # Prepare response
        response = {'top_classes': top_classes, 'probabilities': top_probabilities}
        return response
    except Exception as e:
        return str(e)

