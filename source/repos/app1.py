%%writefile app.py

import streamlit as st
import datetime
import time
from datetime import date	
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objs as go

# import profit -- we will come back
import profit
from prophet import Prophet
from prophet.plot import plot_plotly

st.title("Indian Stock Prediction App")

plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize']=(15,8)

stock = ("INFY.NS", "TCS.NS", "NSEI", "DMART.NS")

selected_stock = st.selectbox("Select stock / Index", stock)

StartDate = st.date_input(
	"StartDate:", datetime.date(2015,1,1)
	)

st.write("My Start date is ", StartDate)

TODAY = date.today().strftime("%Y-%m-%d")

number_of_years = st.slider("Years to predict", 1,3)

period = number_of_years * 365

def load_data(ticker):
	data=yf.download(ticker,StartDate, TODAY)
	data.reset_index(inplace=True)
	return data

stock_data = load_data(selected_stock)
st.write(stock_data.head())
st.write(stock_data.tail())

def plot_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=stock_data['Date'],y=stock_data['Close']))
	fig.layout.update(title_text="Stock time series data", xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_data()

df_train = stock_data[['Date','Close']]

df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

m=Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predic(future)

st.subheader("Forecasted Data")
st.write(forecast.tail())

fig_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_forecast)