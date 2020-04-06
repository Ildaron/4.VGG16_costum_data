from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib as plt
import cv2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle

img_width, img_height = 160, 160

batch_size = 8

model = VGG16(weights="my_vgg16_cat_dogs.h5")



print ("ok1")

img_path = "cat.jpg"
img = image.load_img(img_path, target_size=(160, 160))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

print (features)

