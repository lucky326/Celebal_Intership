from flask import Flask, jsonify, request, send_file
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import cv2
import numpy as np
import io

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

app = Flask(__name__)

# Define dataset paths for the training model
base_dir = 'images' 
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

@app.route('/train', methods=['POST'])
def train_model():
    # Data augmentation and rescaling
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Initialize the generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    # Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=25
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator)
    return jsonify({
        'test_accuracy': accuracy,
        'history': history.history
    })


@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Load the image
    image_path = request.json['landscape.jpg']
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Function to plot histograms
    def plot_histogram(image, color):
        histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))
        return histogram, bin_edges

    # Prepare a bytes buffer to send the plots as images
    buffer = io.BytesIO()

    # Plot histograms for each color channel
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    colors = ('r', 'g', 'b')
    titles = ('Red Channel', 'Green Channel', 'Blue Channel')

    for i, (color, title) in enumerate(zip(colors, titles)):
        histogram, bin_edges = plot_histogram(image_rgb[:, :, i], color)
        axs[i].plot(bin_edges[0:-1], histogram, color=color)
        axs[i].set_xlim(0, 255)
        axs[i].set_title(title)
        axs[i].set_xlabel('Pixel Intensity')
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
