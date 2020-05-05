#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from nsepy import get_history
from datetime import date

import streamlit as st


@st.cache
def loadData():
    symbol = st.text_input("Enter symbol")
    df = get_history(symbol=symbol, start=date(2010,1,1), end=date.today())
    df['Date'] = df.index

    return df

@st.cache
def showMap():
    plotData = loadData()

    st.write(plt.scatter(plotData['Date'], plotData['Close Price']))
    st.write(plt.plot(plotData['Date'], plotData['Close Price']))

def main():

    st.title("NSE Real-Time Stocks Analysis")
    st.header("Enter the symbol of the stock you want to analyse and we will predict the next day Close Price for you")

    loadData()
    showMap()

if __name__ == '__main__':
    main()