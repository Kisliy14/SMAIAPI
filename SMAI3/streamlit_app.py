import streamlit as st
import matplotlib.pyplot as plt
import requests
import json
import pandas as pd

st.sidebar.subheader("Выберите акции")
selected_stocks = st.sidebar.text_input("Введите символы акций через запятую (например, AAPL,MSFT)", value="AAPL")
days = st.sidebar.number_input("Введите количество дней", value=30)
if st.sidebar.button("Построить графики цен на акции"):
    def plot_stock_price(ticker_symbol, days):
        data = yf.download(ticker_symbol, period=f"{days}d")
        plt.figure(figsize=(10, 6))
        plt.plot(data['Adj Close'])
        plt.title(f"Цена акций {ticker_symbol}")
        plt.xlabel("Дата")
        plt.ylabel("Цена (USD)")
        st.pyplot(plt)

    # Отладочный вывод для проверки содержания данных перед отправкой запроса
    st.write("Отладочный вывод данных перед отправкой запроса:", selected_stocks, days)

    # Отправка данных в приложение FastAPI
    url = 'http://127.0.0.1:8000/train_model/'
    headers = {'Content-Type': 'application/json'}
    data = {'stock_symbols': selected_stocks, 'days': days}
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        st.write("Графики цен на акции успешно построены!")
    else:
        st.write("Не удалось построить графики цен на акции. Ошибка:", response.text)

st.set_option('deprecation.showPyplotGlobalUse', False)

get_url = 'http://127.0.0.1:8000/predict_future/'
get_params = {'days': days}
get_response = requests.post(get_url, json=get_params) 

if get_response.status_code == 200:
    future_data = get_response.json()
    for stock, predictions in future_data.items():
        plt.figure(figsize=(16, 6))
        plt.title(f'Прогноз цены закрытия на будущее для {stock}')
        plt.xlabel('Дни', fontsize=18)
        plt.ylabel('Цена закрытия в USD ($)', fontsize=18)
        plt.plot(predictions, label='Будущие прогнозы')
        plt.legend()
        st.pyplot()
else:
    st.write("Не удалось получить данные по акциям с сервера. Ошибка:", get_response.text)

st.title("Визуализация цен на акции")
