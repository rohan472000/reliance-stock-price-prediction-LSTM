import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def predict_stock_price():
    # Read data
    df = pd.read_csv("./RELIANCE_5yrs.csv")

    # Get close price data
    data = df.filter(['Close']).values

    # Scale data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    # Split data into training and testing sets
    training_data_len = int(len(data) * 0.8)
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(10, len(train_data)):
        x_train.append(train_data[i-10:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape data for LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Define LSTM model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(tf.keras.layers.LSTM(100, return_sequences=False))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(1))

    # Compile and fit the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=100)

    # Create testing data set
    test_data = scaled_data[training_data_len - 10:, :]
    x_test = []
    y_test = data[training_data_len:, :]

    for i in range(10, len(test_data)):
        x_test.append(test_data[i-10:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Predict the future trend
    future_data = df.filter(['Close'])
    last_60_days = future_data[-10:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    future_prediction = model.predict(X_test)
    future_prediction = scaler.inverse_transform(future_prediction)
    print('Predicted price for the next day:', future_prediction[0][0])

    # make predictions for the next 10 days
    future_data = df.filter(['Close'])
    last_60_days = future_data[-60:]
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_prices = []
    for i in range(10):
        predicted_price = model.predict(X_test)
        predicted_prices.append(predicted_price[0][0])
        X_test = np.append(X_test, [predicted_price], axis=1)

    # invert the scaling
    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # print the predicted prices
    print(predicted_prices)
    # # Append predicted prices to Excel file
    # today = datetime.now().strftime("%Y-%m-%d")
    # filename = 'predicted_prices.xlsx'
    # if not os.path.isfile(filename):
    #     df_predicted = pd.DataFrame(predicted_prices, columns=['Day 1'])
    #     df_predicted.index = [today]
    #     df_predicted.to_excel(filename)
    # else:
    #     df_predicted = pd.read_excel(filename, index_col=0)
    #     df_predicted['Day ' + str(df_predicted.shape[1] + 1)] = predicted_prices
    #     df_predicted.to_excel(filename)
    # Append predicted prices to Excel file
    today = datetime.now().strftime("%Y-%m-%d")
    filename = 'predicted_prices.xlsx'
    if not os.path.isfile(filename):
        df_predicted = pd.DataFrame(predicted_prices.reshape(1, -1), columns=['Day ' + str(i) for i in range(1, 11)])
        df_predicted.index = [today]
        df_predicted.to_excel(filename)
    else:
        df_predicted = pd.read_excel(filename, index_col=0)
        df_predicted['Day ' + str(df_predicted.shape[1] + 1)] = predicted_prices.reshape(1, -1)
        df_predicted.to_excel(filename)

# calling function
# predict_stock_price()        