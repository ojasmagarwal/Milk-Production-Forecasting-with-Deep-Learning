import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('monthly_milk_production.csv',index_col='Date',parse_dates=True)
df.index.freq='MS'

df.plot(figsize=(12,6))

from statsmodels.tsa.seasonal import seasonal_decompose
     
results = seasonal_decompose(df['Production'])
results.plot();

#training set and test set
train = df.iloc[:156]
test = df.iloc[156:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
     
from keras.preprocessing.sequence import TimeseriesGenerator

# define generator

n_input = 12
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
     
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
     
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
     
print(model.summary())

# fit model
model.fit(generator,epochs=35)

last_train_batch = scaled_train[-12:]
last_train_batch = last_train_batch.reshape((1, n_input, n_features))
model.predict(last_train_batch)

test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    # append the prediction into the array
    test_predictions.append(current_pred) 
    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

print(test_predictions)
true_predictions = scaler.inverse_transform(test_predictions)

test['Predictions'] = true_predictions

test.plot(figsize=(14,5))