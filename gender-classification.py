import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import *
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *

#Dataset is: https://www.kaggle.com/datasets/hb20007/gender-classification

table = pd.read_csv(r"/kaggle/input/gender-classification/Transformed Data Set - Sheet1.csv")

table["Favorite Color"] =  table["Favorite Color"].map({"Cool":0,"Neutral":1,"Warm":2,})
table["Favorite Music Genre"] =  table["Favorite Music Genre"].map({"Electronic":0,"Rock":1,"Hip hop":2,"Folk/Traditional":3,"Jazz/Blues":4,"Pop":5,"R&B and soul":6})
table["Favorite Beverage"] =  table["Favorite Beverage"].map({"Wine":0,"Vodka":1,"Whiskey":2,"Doesn't drink":3,"Beer":4,"Other":5})
table["Favorite Soft Drink"] =  table["Favorite Soft Drink"].map({"7UP/Sprite":0,"Fanta":1,"Coca Cola/Pepsi":2,"Other":3})

table["Gender"] =  table["Gender"].map({"F":0,"M":1})
y=table["Gender"]
x= table.drop(["Gender"],axis=1)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)

model = Sequential()
model.add(Dense(32,input_dim=X_train.shape[1],activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(X_train,y_train,epochs=64,batch_size=32,validation_split=0.1)
loss,accuracy = model.evaluate(X_test,y_test)

test_data=np.array([[2,0,1,2]]) #Example prediction data

#To predict
result= model.predict(test_data)
print(f'Prediction is: {result}')
