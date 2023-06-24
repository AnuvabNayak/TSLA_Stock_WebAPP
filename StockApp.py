import numpy as np
import pandas as pd

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import yfinance as yf
import matplotlib.pyplot as plt
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM
import math
import streamlit as st



st.title('TESLA: Stock Market Prediction')

user_input = st.text_input('Enter Stock Tiker', 'TSLA')
df = yf.download("TSLA",start="2010-01-01", end="2023-05-31", progress=True)

# df = web.DataReader(user_input, 'yf', start, end)

# Describ Data
st.subheader('Data from 2010 - May, 2023')
st.write(df.describe())

# Viszualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize= (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


# create a new data frame with only 'Close column'
data = df.filter(['Close'])
dataset = data.values #convert the data frame to a numpy array
training_data_len = math.ceil(len(dataset)*.8)  # number of rows to train the model on
training_data_len

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

#create the training dataset
#create the scaled training dataset

train_data = scaled_data[0:training_data_len, :]

# #Split the data into x_train, y_train datasets
# x_train = []
# y_train = []
# for i in range(60,len(train_data)):
#     x_train.append(train_data[i-60:i, 0])
#     y_train.append(train_data[i,0])
#     if i<=60:
#         print(x_train)
#         print(y_train)
#         print()


# Load my Model
model = load_model('keras_model.h5')

test_data= scaled_data[training_data_len-60:, :]
#create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
    
#convert the data to a numpy array
x_test = np.array(x_test)

#predicting the data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#get the root mean square error(RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse


# Final Graph


#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visialization of the data
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(16,8))
# plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price' ,fontsize=18)
plt.plot(train['Close'],linewidth=3.5)
plt.plot(valid[['Close','Predictions']],linewidth=3.5)
plt.legend(['Train','Valid','Predictions'], loc='best')
st.pyplot(fig2)