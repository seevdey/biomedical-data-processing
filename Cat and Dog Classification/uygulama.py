# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:54:09 2022

@author: sevde
"""
"""
10000 resim var, 2000 tanesi test, 8000 tanes train

cats, dogs --> 2 tane sınıfımız var --> binary classification --> 
2 şekilde sınıflandırabiliriz 
1.)örneğin 0 kedi, 1 köpek diyebiliriz --> biz bunu kullanacağız
2)ya da arka arkaya 2 tane nöron kullanırım biri köpek diğeri kedileri temsil eder.
""" 

#CNN mimarisi
from keras.models import Sequential
#Sequential() --> veriler birbirini takip ediyor

#Convulation + ReLu
#Conv katmanında filtreleme işlemi yapıyoruz
#from keras.models import Conv2D
from tensorflow.keras.layers import Conv2D

#Max Pooling
#ortaya çıkan features map lerin boyutlarını düşürüyoruz
from keras.layers import MaxPooling2D

#Flatten --> düzleştirme
from keras.layers import Flatten

#Fully Connected
#elimizdeki filtreleri tek boyutlu bir hale getiriyoruz
from keras.layers import Dense
#Dense --> Fully Connected Layer oluşturmama yardımcı oluyor. Tek boyutlu bir vektöre çeviriyor.

#test ve eğitim sonucu birbirinden çok farklıysa ezberlemiş demek oluyo. Dropout bunun önüne geçmek için kullanılıyor
#Dropout ile bazı nöronları çıkarmış oluyoruz. ÖZelliklerinin birbirine bağımlı olmasının önüne geçiyoruz
from keras.layers import Dropout

# CTRL+i ile bilgi alabilirsin

#CNN i başlat
siniflandirici = Sequential()

#CNN layer ekleme
siniflandirici.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
"""
Conv3D --> videolarda işlem yapıyor olsaydık ve zaman da bir parametre olsaydı kullanılırdı
32 --> filtre sayısı
(3,3) --> filtre büyüklüğü --> bize kolaylık sağlayacağı için kare filtrelerle çalışacağız
input shape --> CNN e gelmesini beklediğimiz resimlerin büyüklüğü
64,64,3 --> 64,64 boyut --> 3 derinlik RGB formatında gelsin diye
input shape --> CNN e gelecek olan oluşturmuş olduğumuz mimarinin girdi büyüklüğünü belirtiyor. Belli bir büyüklük belirleyeceğim, bunları mimariye gelmeden önce bu boyuta zorlayacağım. küçükse büyültcem, büyükse küçültcem

activation fonksiyonları -->  relu, tanh, sigmoid
en iyi sonucu verenlerden biri relu
eğer hangisini seçmeniz gerektiğini bilmiyorsanız ara katmanlarda her zaman relu kullanmanız tavsiye edilir

padding in sağladığı avantaj özellik haritamızın birinden diğerine geçerken boyutunun değişmemesini sağlıyor
padding uygulamazsam kullandığım filtrenin büyüklüğüne ve stride a bağlı olarak boyutu giderek küçülmeye başlıyor. Bunun önüne geçmek istiyorsam herhangi bir değer yazabilirim

Deney yaparken kontrollü deney yapman gerekiyor. Filtre sayısı ve filtre boyutunu değiştirmek istersem aynı anda değil farklı farklı değiştirip sonuçlara bakman gerek.

Bir sonraki katmanın derinliği farklı oluyor
"""

# Max Pooling uygulama
siniflandirici.add(MaxPooling2D(2,2)) #2D olarak uygulamasını istemişim yukarda #Resmin boyutunda yarı yarıya azalma olacak
#MaxPooling2D parametresi --> pool_size --> kaça kaçlık bir düşürme uygulayacağım
#MaxPooling2D --> özellik haritasının boyutunu düşürecek. kaça kaç boyutta uyguladıysak o boyutta düşürecek 
#2,2 lik bir alanda bir piksel değeri üretiyo. Bu piksel değeri en büyük olanı seçiyo

"""
filtrenin üzerindeki değerleri ağırlıklar gibi düşünebilirsin
max pooling de eğitilmesi gereken parametre sayısı 0 çıktı --> max pooling katmanında yaptığım ttek şey özellik haritamın 
boyutunu küçültmek, burda herhangi bir eğiteceğim parametre olmadığı için 0 çıktı. bu yüzden ne derinlikte ne 
parametre sayısında değişiklik olmuyor'
"""

#resmi sadece ilk katmanda kullanıyoruz. Sonraki katmanlarda features map ile işlem yapıyoruz

# ikinci CNN katmanı ekleme

#2. CNN eklerken tekrardan input shape kullanmana gerek yok pc onu senin için ilk katmanda hesaplıyıp veriyor 2. katmana


siniflandirici.add(Flatten())
#Flatten --> düzleştirme --> artık tam bağlı katmana hazır hale getirmeye başlıycaz
#eğitmeyle ilgili herhangi bir parametre yok. parametre sayısında değişiklik yok

siniflandirici.add(Dense(units=128, activation='relu')) #kaç tane nöron olması gerektiği ve hangi aktivasyon fonk kullanacağını belirledik
#units i nöron sayısı gibi düşünebilirsin. Araya katman ekliyoruz
#Dense --> tam bağlı katman mimarisine sahip --> YSA'daki gibi her girdi bir sonraki katmandaki nörona tam bağlı olacak

siniflandirici.add(Dense(units=1, activation='sigmoid'))
#128 tane girdi var + 1 bias değeri = 129

siniflandirici.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#loss önemli bir parametre --> tek bir sınıflandırma yapıyorsan binary_crossentropy
#oluşturduğum mimariye dikkat etmem gerekiyor. son katmanda tek 1 tane nöron bıraktıysam binary sınıflandırmaya çalışıyorsam binary_crossentropy kullanacağım
#????eğer son katmanım çoklu bir yapıya sahipse multi classification yapıyorsa bununla ilgili optimizer ı kullanmam gerekiyor.
#metrics=['accuracy'] --> metrics olarak doğruluğuna bakıyoruz

siniflandirici.summary()
#summary fonksiyonu --> bizim oluşturduğumuz mimari ile ilgili bütün parametreleri veriyor
"""
total params : bu ağdaki toplam parametre sayısı
trainable params : eğitilebilir parametre sayısı
non-trainable params : eğitilemez parametre sayısı
"""

#başarı oranı düşükse ilk bakacağımız şey etiketlemeyi doğru yapılıp yapılmadığını kontrol etmek

#resimler aynı boyutta değil, bunların boyutlarını düzenlememiz gerekiyor

# Data Augmentation --> elimdeki veri sayısını artırma yöntemleri

from keras.preprocessing.image import ImageDataGenerator #yeni resim üretmeme yardımcı olacak
#ezberlemenin önüne geçmek için diğer bir yöntem elimdeki veri seti sayısını arttırmak 
#overfitting i önlemek verisetimizi büyütmek için uyguluyoruz


#eğitim veri setine uygulayacağım değer
train_datagen = ImageDataGenerator(
    rescale = 1./255, #rescale; özellik ölçeklendirme yapıyor. 0-255 arasına çekiyor. Daha küçük sayılarla işlem yapıyoruz
    shear_range = 0.2, #kırpma
    zoom_range = 0.2,
    horizontal_flip = True
    )

#test veri setine uygulayacağım değer
#bunu eğitim için yapıyorsam test için de yapmam gerekiyor --> rescale
#çünkü test verisini artırmak gibi bir amacı yok
test_datagen = ImageDataGenerator(
    rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', #dosya yolu
                                           target_size = (64,64), #gelecek resimleri 64,64 e çekecek (zorlayacak kısım burası)
                                           #çünkü biz CNN oluştururken 64,64 oluşturmasını istedik
                                           # #tüm resimler 64 64 boyutuna dönüştürülecek
                                           batch_size = 32, #her 32 adımda bir filtreleri güncelle
                                           #,#her turda kaç adım atmam gerektiği. burdaki 32 den batch den geliyor
                             #örneklem sayısı/batch_size
                                           class_mode = 'binary')
#flow_from_directory --> klasörlerin içerisinden bunları alıcam ve benim belirlediğim hedefe getirecek

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                          target_size = (64,64),
                                          batch_size = 64, #her 64 adımda bir filtreleri güncelle
                                          class_mode = 'binary')
"""
#ÇIKTI
Found 8000 images belonging to 2 classes.
Found 2000 images belonging to 2 classes.
"""

#keras otomatik olarak kendi sınıflandırma işlemini yapıyo


siniflandirici.fit_generator(training_set, #eğitim seti
                             steps_per_epoch = 8000//32, #her adımda(step) uygulanacak epoch size
                             epochs = 5,
                             validation_data = test_set,
                             validation_steps = 2000//32)
#250 değerinin olmasının sebebi 8000 veri var ve her 32 adımda bir güncelleme yapılacak şekilde ayarladık. Yani 250 kere ağırlık güncelleniyor

"""
#VERİLERİ KAYDETME
siniflandirici.save_weights('agirliklar_2022.h5') #ağırlıkları saklamak

siniflandirici.load_weights('agirliklar_2022.h5')

siniflandirici.save('model2022') #modeli saklamak

from keras.models import load_model
siniflandirici2 = load_model(('model2022'))
"""

import pandas as pd
test_set.reset
ytesthat = siniflandirici.predict_generator(test_set)
df = pd.DataFrame({
    'filename':test_set.filenames,
    'predict':ytesthat[:,0],
    'y':test_set.classes
})

pd.set_option('display.float_format', lambda x: '%.5f' % x)
df['y_pred'] = df['predict']>0.5
df.y_pred = df.y_pred.astype(int)
df.head(10)


#Prediction of test set
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix = confusion_matrix(df.y,df.y_pred)
sns.heatmap(conf_matrix,cmap="YlGnBu",annot=True,fmt='g');
plt.xlabel('predicted value')
plt.ylabel('true value');

# Model Accuracy
x1 = siniflandirici.evaluate_generator(training_set)
x2 = siniflandirici.evaluate_generator(test_set)

print('Training Accuracy  : %1.2f%%     Training loss  : %1.6f'%(x1[1]*100,x1[0]))
print('Validation Accuracy: %1.2f%%     Validation loss: %1.6f'%(x2[1]*100,x2[0]))