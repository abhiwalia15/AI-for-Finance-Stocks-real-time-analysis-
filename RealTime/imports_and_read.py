#Import the libraries
#import math
#import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
# import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')
from keras.models import load_model
from nsepy import get_history
from datetime import date
import datetime
import streamlit as st

from PIL import Image
