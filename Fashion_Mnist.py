#Kütüphaneleri yükle
#import keras
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

#Veri setimizi yükleyelim
fashion_mnist=keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

#Veri setinin boyutları
#Eğitim setinin boyutu
print(train_images.shape)
#Test setinin boyutu
print(test_images.shape)

#Modelimizi oluşturalım
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten


model=Sequential()

#1. Convolutional layer
model.add(Conv2D(64,kernel_size=3,activation="relu",input_shape=(28,28,1)))
# 2. Convolutional Layer
model.add(Conv2D(32,kernel_size=3,activation="relu"))

#Flatten Layer
model.add(Flatten())

#Tam Bağlı Katman 
model.add(Dense(10,activation="softmax"))

#Modelimizi inceleyelim.
model.summary()

#Modeli derle
import tensorflow as tf
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

train_images=train_images.reshape(-1,28,28,1)
test_images=test_images.reshape(-1,28,28,1)

#Özellik ölçeklendirme yapalım
train_images=train_images/255.0
test_images=test_images/255.0

#Önce modeli eğitip sonra test etme
model.fit(train_images,train_labels,epochs=10)
test_loss,test_accuracy=model.evaluate(test_images,test_labels)


model.load_weights("agirliklar2022.h5")

predictions=model.predict(test_images)

import numpy as np
tahmin_sinif=np.argmax(predictions,axis=1)
import sklearn.metrics as metrics
print("Accuracy=", metrics.accuracy_score(test_labels,tahmin_sinif))
print(metrics.confusion_matrix(test_labels,tahmin_sinif))





























