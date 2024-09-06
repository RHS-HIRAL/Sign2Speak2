import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocess import load_dataset

def create_model(input_shape):
    model = tf.keras.Sequential([
        # Normalize pixel values automatically within the model
        tf.keras.layers.Rescaling(1./255, input_shape=input_shape),
        
        # First convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        
        # Second convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        
        # Third convolutional layer
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        
        # Flatten and fully connected layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Prevent overfitting
        
        # Output layer for 28 classes (A-Z, space, nothing)
        tf.keras.layers.Dense(28, activation='softmax')
    ])
    
    # Compile the model with optimizer, loss, and metrics
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    dataset_path = r'C:\Hiral\Sign2Speak\dataset'
    
    # Load dataset and preprocess images
    data, labels = load_dataset(dataset_path)
    
    # Ensure data is reshaped and normalized in the load_dataset function or here
    # For grayscale images, the shape should be (64, 64, 1)
    # data = data / 255.0  # If not already normalized
    
    # Encode the labels as integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)
    
    # Define the input shape for the model
    input_shape = (64, 64, 1)  # Grayscale image
    
    # Create the model
    model = create_model(input_shape)
    
    # Print model summary
    model.summary()
    
    # Early stopping and learning rate scheduler callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

    # Data augmentation for the training set
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    
    # Train the model with data augmentation
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        validation_data=(X_val, y_val),
                        epochs=20,  # Increased epochs for better training
                        callbacks=[early_stopping, lr_scheduler])
    
    # Save the trained model
    model.save('trained_model.keras')
    print("Model trained and saved to 'trained_model.keras'")
