from fastapi import FastAPI, HTTPException
from typing import List, Dict
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import yfinance as yf
import pickle
import json
import uvicorn

app = FastAPI()

class ItemStorage:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0,1))

    def train_model(self, value_variable: str, days: int):
        # self.stocks = [s.strip().upper() for s in value_variable.split(',')]
        # if not self.stocks:
        #     raise HTTPException(status_code=400, detail="No stocks provided")
            
        # plots = {}
        # for stock in self.stocks:
        #     plots[stock] = self.plot_stock_price(stock, days)
            
        end = datetime.now()
        start = datetime(end.year - 1, end.month, end.day)
        tech_list = []
        tech_list.extend(value_variable)
        df_list = []
        for stock in tech_list:
               df_list.append(yf.download(stock, start, end))
        df = pd.concat(df_list, axis=1, keys=tech_list)
        data = df.filter(['Close'])
        dataset = data.values
        training_data_len = int(np.ceil(len(dataset) * .95 ))
            
        scaled_data = self.scaler.fit_transform(dataset)
        train_data = scaled_data[0:int(training_data_len), :]
            
        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
                x_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        #self.model = pickle.load(open('C:/Users/Kisliy/Desktop/SMAI3/models/model.pkl','rb'))
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dense(units=25))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(x_train, y_train, batch_size=1, epochs=2)

    def predict_future(self, days: int) -> Dict[str, List[float]]:
        # Ваш код предсказания будущих цен
        if self.model is None:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        else:
            end = datetime.now()
            start = end - pd.DateOffset(days=60)
            future_dates = pd.date_range(end, periods=days+1, freq='B')[1:]
            x_future = []
            for stock in self.stocks:
                df = yf.download(stock, start, end)
                data = df.filter(['Close'])
                dataset = data.values
                scaled_data = self.scaler.transform(dataset)
                last_60_days = scaled_data[-60:]
                for i in range(60, len(last_60_days)):
                    x_future.append(last_60_days[i-60:i, 0])

            x_future = np.array(x_future)
            x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))
            future_predictions = self.model.predict(x_future)
            future_predictions = self.scaler.inverse_transform(future_predictions)

            future_dict = {}
            for i, stock in enumerate(self.stocks):
                future_dict[stock] = future_predictions[i*days:(i+1)*days].flatten().tolist()

            return future_dict

storage = ItemStorage()

@app.post("/train_model/")
async def train_model(stock_symbols: str, days: int):
    storage.train_model(stock_symbols, days)
    return {"message": "Model trained successfully"}

@app.post("/predict_future/")
async def predict_future(days: int):
    future_data = storage.predict_future(days)
    return future_data

# Запуск сервера
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
