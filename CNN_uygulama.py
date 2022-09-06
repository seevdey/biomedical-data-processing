
#CNN Mimarimiz
from keras.models import Sequential

#Convolution +ReLU
from keras.layers import Conv2D

#Max Pooling
from keras.layers import MaxPooling2D

#Flatten
from keras.layers import Flatten

#Fully Connected
from keras.layers import Dense

#CNN i baslat
siniflandirici=Sequential()

#CNN layer ekleme
siniflandirici.add(Conv2D(32,(3,3),
                          input_shape=(64,64,3),
                          activation='relu'))

# 32 filtre sayımız
# (3,3) filtremin büyüklüğü 
# input shape CNN e gelmesini beklediğimiz resimlerin büyüklüğü
#activation relu, tanh, sigmoid

# Max Pooling uygulama

siniflandirici.add(MaxPooling2D(2,2))
# ikinci CNN katmanı ekleme


siniflandirici.add(Flatten())

siniflandirici.add(Dense(units=128,activation='relu'))

siniflandirici.add(Dense(units=1, activation='sigmoid'))

siniflandirici.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

siniflandirici.summary()

# Data Augmentation

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )
test_datagen=ImageDataGenerator(
    rescale=1./255)

training_set=train_datagen.flow_from_directory('dataset/training_set',
                                           target_size=(64,64),
                                           batch_size=32,
                                           class_mode='binary')

test_set=test_datagen.flow_from_directory('dataset/test_set',
                                          target_size=(64,64),
                                          batch_size=64,
                                          class_mode='binary')

siniflandirici.fit_generator(training_set,
                             steps_per_epoch=8000//32,
                             epochs=5,
                             validation_data=test_set,
                             validation_steps=2000//32)

siniflandirici.save_weights('agirliklar_2022.h5')

siniflandirici.load_weights('agirliklar_2022.h5')

siniflandirici.save('model2022')

from keras.models import load_model
siniflandirici2=load_model(('model2022'))


































