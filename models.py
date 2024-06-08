import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import Model
import numpy as np

def create_and_train_model(train_dir, input_shape, num_categories, model_type):
    # Define input shape and number of classes

    # Create model
    if model_type == 'custom':
        model = create_custom_model(input_shape, num_categories)
    elif model_type == 'resnet50':
        model = create_resnet50_model(input_shape, num_categories)
    elif model_type == 'vgg16':
        model = create_vgg16_model(input_shape, num_categories)

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Define image data generators
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


    # Set batch size
    batch_size = 32

    # Generate batches of tensor image data for training and validation
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')  # set as training data

    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')  # set as validation data
    
    # Train the model
    try:
        model.fit(
            train_generator,
            epochs=10,
            validation_data=validation_generator)
        print("Training completed successfully.")
    except Exception as e:
        print("Error occurred during training:", str(e))


    test_loss, test_acc = model.evaluate(train_generator,  verbose=2)

    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)
    return model

def count_categories(train_dir):
    class_names = []
    for root, dirs, files in os.walk(train_dir):
        for dir in dirs:
            class_names.append(dir)
    num_categories = len(class_names)
    return num_categories, class_names

# Function to create custom model
def create_custom_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Function to create ResNet50 model
def create_resnet50_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    print("resnet50 created")
    return model

# Function to create VGG16 model
def create_vgg16_model(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    print("VGG16 created")
    return model

def load_synset(synset_path):
    """Load class names from the synset file."""
    class_names = []
    with open(synset_path, 'r') as f:
        for line in f:
            class_names.append(line.strip())
    return class_names
