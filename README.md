# Cats-and-Dogs-images-classification-using-CNN

This project trains a Convolutional Neural Network (CNN) to classify images as cat or dog. It uses the well-known Dogs vs Cats dataset from Kaggle and implements best practices like data augmentation, dropout, and early stopping to reduce overfitting and improve performance.

<h2> Dataset: </h2>

Dataset Source: Kaggle – Dogs vs Cats

The dataset is downloaded with: 
import opendatasets as od
od.download("https://www.kaggle.com/datasets/salader/dogs-vs-cats")

<h2>Dataset structure:</h2>
dogs-vs-cats/
├── train/ (20,000 images)
└── test/ (5,000 images)
Each folder contains mixed cat & dog images.

<h3>Technologies Used:</h3> 

- Python:	Base language
- TensorFlow / Keras:	Deep learning framework
- OpenDatasets:	Download Kaggle dataset
- Matplotlib:	Visualization
- NumPy:	Numeric processing

<h3> Data Preprocessing</h3>

- Image size: 256 × 256
- Batch size: 32
- Pixel normalization: 0–1 scale
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=30,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

<h3>Model Architecture:</h3>
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
MaxPooling2D(2,2),
Dropout(0.25),

Conv2D(64, (3,3), activation='relu'),
MaxPooling2D(2,2),
Dropout(0.25),

Conv2D(128, (3,3), activation='relu'),
MaxPooling2D(2,2),
Dropout(0.25),

Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(1, activation='sigmoid')
])


<h3>Training the Model:</h3>

from tensorflow.keras.callbacks import EarlyStopping

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(train_data, validation_data=test_data, epochs=15, callbacks=[early_stop])

<h3>Prediction Example: </h3>
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img("sample.jpg", target_size=(256,256))
x = image.img_to_array(img)/255
y_pred = model.predict(x.reshape(1,256,256,3))
print("Dog" if y_pred > 0.5 else "Cat")

<h3>Save Model:</h3>
model.save("cat_vs_dog_model.keras")

<h3>Future Improvements:</h3>

1. Transfer Learning (VGG16 / ResNet50 / MobileNetV2)

2. Add confusion matrix visualization

3. Build a Streamlit or Flask web app

4. Hyperparameter tuning (learning rate, batch size)

<h3>How to Run:</h3>

- pip install tensorflow opendatasets matplotlib numpy.

- Run the notebook step-by-step or execute code in Google Colab.

