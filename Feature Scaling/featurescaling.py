# -*- coding: utf-8 -*-
"""

@author: sevde

"""
#Veri Ön İşleme

#1. Adım -- Gerekli Kütüphaneleri Yükle
#numpy matematiksel araçları içerir.
import numpy as np

#matplotlib grafik oluşturmamıza yardımcı olur.
import matplotlib.pyplot as plt

#pandas veri setini yuklemek icin ve kontrol etmek için kullanılır.
import pandas as pd

import sklearn
print(sklearn.__version__)


#2. Adım -- Veri Setini yükle
dataset=pd.read_csv("Data.csv")

X=dataset.iloc[:,:-1].values

print(X)

y=dataset.iloc[:,-1].values

print(y)


#3. Adım -- Eksik Verileri Tamamla

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#print(X)


#4. Adım -- Kategorik Verileri Düzenleme

from sklearn.preprocessing import LabelEncoder

labelencoder_X=LabelEncoder()

X[:,0]=labelencoder_X.fit_transform(X[:,0])

#print(X)

#Bir kolon yerine her kategori için yeni bir kolon oluştur.(Dumy Encoding)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X) #.toarray()

print(X)

#Çıktı içinde encoding yapıyoruz.

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)


#5.Adım -- VeriSetini Eğitim ve Test olarak böl

from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#6. Adım -- Özellik Ölçeklendirme (Feature Scaling)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)









