
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib as plt
import cv2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,MaxPooling2D
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
# Количество эпох
epochs = 1
# Размер выборки
batch_size = 8

base_model = VGG16(weights="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
             include_top=False,
                  input_shape=(160, 160, 3))
#base_model(summary)
#exit(0)
print ("ok1")
#vgg16_weights.h5

train_dir = "train"
validation_dir = "validation"
 
x = base_model.output
x = MaxPooling2D()(x)
x = Dense(1024, activation='relu')(x)
#x = Dense(2048, activation='relu')(x)

print ("ok2")
predictions = Dense(2, activation='softmax')(x)

print ("ok3")

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print ("ok4")
# create generator for train
datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    train_dir,  # train_data_dir
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode= 'categorical')
print ("ok5")
# create generator with validation
validation_generator = datagen.flow_from_directory(
    validation_dir,  #  validation_data_dir
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode= 'categorical')
print ("ok6")

Early_Stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
print ("ok7")

#Use generator
model.fit_generator(
    train_generator,
    steps_per_epoch=17,  #   здесь и 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps = 17,  # здесь
    verbose=1,
    callbacks=[Early_Stopping]
    )


model_json = model.to_json()
json_file = open("my_vgg16_cat_dogs.json", "w")

#json_file.write(model_json)
#json_file.close()


model.save_weights("my_vgg16_cat_dogs.h5")


#with open('model1.json', 'w') as f:
#    f.write(model.to_json())


test_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode= 'categorical')

print (test_generator)

scores = model.evaluate_generator(test_generator, 17  // batch_size)
print("accuracy: %.2f%%" % (scores[1]*100))









