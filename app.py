from flask import Flask, render_template, request, jsonify
import os
import shutil
import threading
import time
import loadImages
import model
import inference

app = Flask(__name__)

def clear_and_load_data():
    try:
        shutil.rmtree('ISIC_Images')
        os.mkdir('ISIC_Images')
        loadImages.main()  # Load new data
        print("Data cleared and new data loaded successfully")
    except Exception as e:
        print("Error:", str(e))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Save uploaded file
        file_path = 'uploaded_image.jpg'
        file.save(file_path)
        
        # Perform image classification
        predicted_class = inference.predict_from_file("medical_model.keras", "synset.txt", file_path)
        
        # Delete the uploaded image
        os.remove(file_path)
        
        # Prepare response
        #response = {'predicted_class': predicted_class}
        response = predicted_class
        print(response)
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

def clear_and_load_data(numPictures):
    try:
        shutil.rmtree('ISIC_Images')
        os.mkdir('ISIC_Images')
        loadImages.main(numPictures)  # Load new data
        print("Data cleared and new data loaded successfully")
    except Exception as e:
        print("Error:", str(e))

@app.route('/clear_and_load_data', methods=['POST'])
def clear_and_load_data_route():
    numPictures = request.form.get('numPictures')
    if numPictures:
        # Create a new thread to avoid blocking the main thread
        clear_and_load_data(numPictures)
        return jsonify({'message': f'Data clearing and loading of {numPictures} pictures done.'})
    else:
        return jsonify({'error': 'No number of pictures provided.'})

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    try:
        # Count categories
        num_categories, class_names = model.count_categories('ISIC_Images')

        # Save the class names to a file
        with open('synset.txt', 'w') as f:
            for name in class_names:
                f.write(name + '\n')
        
        # Call train_model to retrain the model
        retrained_model = model.train_model(train_dir='ISIC_Images', input_shape=(224, 224, 3), num_classes=num_categories)
        
        # Save the retrained model
        retrained_model.save('medical_model.keras')
        
        return jsonify({'message': 'Model retraining and saving completed successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
