import streamlit as st
from PIL import Image
from nsepy import get_history
from datetime import date
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model


def main():
    
    image = Image.open('sm.jpeg')
    st.image(image,use_column_width=True)

    st.title("NSE Real-Time Stocks Analysis and Predictions")
    
    st.header("Select the stock and check its next day predicted value")
    
    choose_stock = st.sidebar.selectbox("Choose the Stock!",["NONE","Reliance", "PowerMech Solns.", 'RepcoHomes'])

    if(choose_stock == "Reliance"):

        # get abfrl real time stock price
        df1 = get_history(symbol='reliance', start=date(2010,1,1), end=date.today())
        df1['Date'] = df1.index

        st.header("Reliance India NSE Last 5 Days DataFrame:")
        
        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df1.tail())

        ## Predictions and adding it to Dashboard
        new_close_col = df1.filter(['Close'])
        mm_scale = MinMaxScaler(feature_range=(0, 1))
        mm_scale_data = mm_scale.fit_transform(new_close_col)
        new_close_col_val = new_close_col[-30:].values
        new_close_col_val_scale = mm_scale.transform(new_close_col_val)

        X_test = []
        X_test.append(new_close_col_val_scale)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        network = load_model("reliance.model")
        new_preds = network.predict(X_test)
        new_preds = mm_scale.inverse_transform(new_preds)
        #print(new_preds[0])
        
        # next day
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(new_preds[0][0])

        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df1[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High','Low']])

# driver code
if __name__ == '__main__':
    main()