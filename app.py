from flask import Flask, render_template, request, jsonify
import os
import shutil
import loadImages
from models import create_and_train_model, count_categories
import inference
from werkzeug.utils import secure_filename

app = Flask(__name__)

#inference.load_model("medical_model.keras")

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
        

        model_type = request.form.get('selectedModelTypePredict')

        # Ensure the uploads folder exists
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        # Save the uploaded file to the uploads folder
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Save the retrained model with different names based on model_type
        if model_type == 'custom':
            model_name = "custom_model.keras"
        elif model_type == 'resnet50':
            model_name = "resnet50_model.keras"
        elif model_type == 'vgg16':
            model_name = "mobilenetv2_model.keras"
        else:
            model_name = "custom_model.keras"

        print("choosen model: " + model_type)
        # Perform image classification
        predicted_class = inference.predict_from_file(model_name, "synset.txt", file_path)
        # Delete the uploaded image
        os.remove(file_path)
        
        # Prepare response
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
        data = request.get_json()
        model_type = data.get('model_type', 'custom')
        # Count categories
        num_categories, category_names = count_categories('ISIC_Images')

        # Save the class names to a file
        with open('synset.txt', 'w') as f:
            for name in category_names:
                f.write(name + '\n')

        # Call train_model to retrain the model
        print("choosen model: " + model_type)
        model = create_and_train_model(train_dir='ISIC_Images', input_shape=(224, 224, 3), num_categories=num_categories, model_type=model_type)


        # Save the retrained model with different names based on model_type
        if model_type == 'custom':
            model.save('custom_model.keras')
        elif model_type == 'resnet50':
            model.save('resnet50_model.keras')
        elif model_type == 'vgg16':
            model.save('mobilenetv2_model.keras')
        else:
            model.save('custom_model.keras')
        
        return jsonify({'message': 'Model retraining and saving completed successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
