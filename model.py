import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def create_resnet_model(input_shape, num_classes):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(train_dir, input_shape, num_classes, batch_size=32, epochs=10):
    # Create and compile the model
    model = create_resnet_model(input_shape, num_classes)
    
    # Create TensorFlow datasets
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=input_shape[:2],
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='training'
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=input_shape[:2],
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='validation'
    )

    # Fit the model
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    return model

def count_categories(train_dir):
    class_names = []
    for root, dirs, files in os.walk(train_dir):
        for dir in dirs:
            class_names.append(dir)
    num_categories = len(class_names)
    return num_categories, class_names