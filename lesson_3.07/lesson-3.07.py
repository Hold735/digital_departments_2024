"""
Конспект занятия № 3.07
"Нейронные сети"
"""

"""
Нейронные сети:
- сверточные (конволюционные) нейронные сети (CNN) - компьютерное зрение, классификация изображений
- рекурентные нейронные сети (RNN) - распознавание рукописного текста, обработка естественного текста
- генеративные состязательные сети (GAN) - создание художественных, музыкальных произведений
- многослойный перцептрон - простейший тип НС
"""
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

img_path = "./lesson_3.07/data/dog.png"
img = image.load(img_path, target_size=(224,224))

import numpy as np

img_array = image.img_to_array(img)
print(img_array.shape)

img_batch = np.expand_dims(img_array, axis=0)

from tensorflow.keras.applications.resnet50 import preprocessing_input

img_processed = preprocessing_input(img_batch)

from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50()

prediction = model.predict(img_processed)

from tensorflow.keras.applications.resnet50 import decode_predictions

print(decode_predictions(prediction, top=5)[0])

# plt.imshow(img)
# plt.show()

TRAIN_DATA_DIR = "./lesson_3.07/data/train_data/"
VALIDATION_DATA_DIR = "./lesson_3.07/data/val_data/"
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
NUM_CLASSES = 2
IMG_WIDTH, IMG_HIGHT = 224, 224
BATCH_SIZE = 64

from tensorflow.keras.applications.mobilenet import preprocess_input, MobileNet

train_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocessing_input,
    rotation_range=20,
    width_shift_range=0.2,
    heigth_shift_range=0.2,
    zoom_range=0.2,
)

val_datagen = image.ImageDataGenerator (preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=12345,
    class_mode="categorial",
)

val_generator = val_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode="categorial",
)

from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.model import Model

def model_marker():
    base_model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HIGHT, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False

    input = Input(shape=(IMG_WIDTH, IMG_HIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation="relu")(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    prediction = Dense(NUM_CLASSES, activation="softmax")(custom_model)
    return Model(inputs=input, output=prediction)

from tensorflow.keras.optimizer import Adam

model = model_marker()
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(),
    metrics=["acc"]
)

import math
num_steps = math.cell(float(TRAIN_SAMPLES) / BATCH_SIZE)

model.fit(
    train_generator,
    steps_per_epoch=num_steps,
    epochs=10,
    validation_data=val_generator,
    validation_steps=num_steps,
)

print(val_generator.class_indices)

model.save("./lesson_3.07/data/model.h5")

#------------------------------------------------------------------

from keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("./lesson_3.07/data/model.h5")
img_path = "./lesson_3.07/data/cat.png"
img = image.load_img (img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

img_processed = preprocessing_input(img_batch)

prediction = model.predict(img_processed)
print(prediction)
