from imports_and_read import * 
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

def main():
    
    image = Image.open('sm.jpeg')
    st.image(image,use_column_width=True)

    st.title("NSE Real-Time Stocks Analysis and Predictions")
    
    st.header("Select the stock and check its next day predicted value")
    
    st.subheader("This study is mainly confined to the s tock market behavior and is \
                intended to devise certain techniques for investors to make reasonable\
                returns from investments .")
    
    st.subheader("Though there were a number of studies , which\
                deal with analysis of stock price behaviours , the use of control chart\
                techniques and fai lure time analysis would be new to the investors. The\
                concept of stock price elast icity,\
                introduced in this study, will be a good\
                tool to measure the sensitivity of stock price movements.")
    
    st.subheader("In this study, \
                 Predictions for the close price is suggested for the National Stock Exchange index,\
                Nifty,\
                based on Long Short Term Based (LSTM)\
                method.") 
    
    st.subheader("We make predictions based on the last 30 days Closing price data\
                which we fetch from NSE India website in realtime.")
    
    st.markdown("Note: This is just a fun project, No one can predict the\
         stock market as of today because there are a\
        lot of factors which needs to be considered\
             before makaing any investments, especially in StockMarket.\
            So it is advisable now to indulge in any\
                 bad decisions based on the predictions shown here.")

    st.header("THANKS FOLKS!!")
    
    st.subheader("Happy Learning")

    st.subheader("Creator: MRINAL WALIA😈😈😈")


    # Stock Section
    choose_stock = st.sidebar.selectbox("Choose the Stock!",
        ["NONE","Aditya Birla Fashion Retail Ltd.", "PowerMech Solns.", 'RepcoHomes', 'IndiaBulls HSG', 'INOX Leisure', 'SpiceJet', 'TataMotors'])

    if(choose_stock == "Aditya Birla Fashion Retail Ltd."):

        # get abfrl real time stock price
        df1 = get_history(symbol='abfrl', start=date(2010,1,1), end=date.today())
        df1['Date'] = df1.index

        st.header("Aditya Birla Fashion NSE Last 5 Days DataFrame:")
        
        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df1.tail())
        
        ## Predictions and adding it to Dashboard
        #Create a new dataframe
        new_df = df1.filter(['Close'])
        #Scale the all of the data to be values between 0 and 1 
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(new_df)
        #Get teh last 30 day closing price 
        last_30_days = new_df[-30:].values
        #Scale the data to be values between 0 and 1
        last_30_days_scaled = scaler.transform(last_30_days)
        #Create an empty list
        X_test = []
        #Append teh past 1 days
        X_test.append(last_30_days_scaled)
        #Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        #load the model
        model = load_model("abfrl.model")
        #Get the predicted scaled price
        pred_price = model.predict(X_test)
        #undo the scaling 
        pred_price = scaler.inverse_transform(pred_price)
        
        # next day
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price)

        ##visualizations
        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df1[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High','Low']])

    elif(choose_stock == "PowerMech Solns."):

        # get abfrl real time stock price
        df2 = get_history(symbol='powermech', start=date(2010,1,1), end=date.today())
        df2['Date'] = df2.index
        
        st.header("PowerMech Soln. NSE Last 5 Days DataFrame:")

        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df2.tail())

        ## Predictions and adding it to Dashboard
        #Create a new dataframe
        new_df = df2.filter(['Close'])
        #Scale the all of the data to be values between 0 and 1 
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(new_df)
        #Get teh last 30 day closing price 
        last_30_days = new_df[-30:].values
        #Scale the data to be values between 0 and 1
        last_30_days_scaled = scaler.transform(last_30_days)
        #Create an empty list
        X_test = []
        #Append teh past 1 days
        X_test.append(last_30_days_scaled)
        #Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        #load the model
        model = load_model("powermech.model")
        #Get the predicted scaled price
        pred_price = model.predict(X_test)
        #undo the scaling 
        pred_price = scaler.inverse_transform(pred_price)
        
        # next day
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price)

        ##visualizations
        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df2['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df2[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df2[['High','Low']])

    
    elif(choose_stock == "IndiaBulls HSG"):

        # get abfrl real time stock price
        df3 = get_history(symbol='ibulhsgfin', start=date(2010,1,1), end=date.today())
        df3['Date'] = df3.index

        st.header("IndiaBulls HSG NSE Last 5 Days DataFrame:")
        
        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df3.tail())
        
        ## Predictions and adding it to Dashboard
        #Create a new dataframe
        new_df = df3.filter(['Close'])
        #Scale the all of the data to be values between 0 and 1 
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(new_df)
        #Get teh last 30 day closing price 
        last_30_days = new_df[-30:].values
        #Scale the data to be values between 0 and 1
        last_30_days_scaled = scaler.transform(last_30_days)
        #Create an empty list
        X_test = []
        #Append teh past 1 days
        X_test.append(last_30_days_scaled)
        #Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        #load the model
        model = load_model("ibulhsgfin.model")
        #Get the predicted scaled price
        pred_price = model.predict(X_test)
        #undo the scaling 
        pred_price = scaler.inverse_transform(pred_price)
        
        # next day
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price)

        ##visualizations
        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df3['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df3[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df3[['High','Low']])

    elif(choose_stock == "INOX Leisure"):

        # get abfrl real time stock price
        df4 = get_history(symbol='inoxleisur', start=date(2010,1,1), end=date.today())
        df4['Date'] = df4.index

        st.header("INOX Leisure NSE Last 5 Days DataFrame:")
        
        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df4.tail())
        
        ## Predictions and adding it to Dashboard
        #Create a new dataframe
        new_df = df4.filter(['Close'])
        #Scale the all of the data to be values between 0 and 1 
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(new_df)
        #Get teh last 30 day closing price 
        last_30_days = new_df[-30:].values
        #Scale the data to be values between 0 and 1
        last_30_days_scaled = scaler.transform(last_30_days)
        #Create an empty list
        X_test = []
        #Append teh past 1 days
        X_test.append(last_30_days_scaled)
        #Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        #load the model
        model = load_model("inoxleisur.model")
        #Get the predicted scaled price
        pred_price = model.predict(X_test)
        #undo the scaling 
        pred_price = scaler.inverse_transform(pred_price)
        
        # next day
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price)

        ##visualizations
        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df4['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df4[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df4[['High','Low']])

    elif(choose_stock == "RepcoHomes"):

        # get abfrl real time stock price
        df5 = get_history(symbol='repcohome', start=date(2010,1,1), end=date.today())
        df5['Date'] = df5.index

        st.header("RepcoHomes NSE Last 5 Days DataFrame:")
        
        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df5.tail())
        
        ## Predictions and adding it to Dashboard
        #Create a new dataframe
        new_df = df5.filter(['Close'])
        #Scale the all of the data to be values between 0 and 1 
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(new_df)
        #Get teh last 30 day closing price 
        last_30_days = new_df[-30:].values
        #Scale the data to be values between 0 and 1
        last_30_days_scaled = scaler.transform(last_30_days)
        #Create an empty list
        X_test = []
        #Append teh past 1 days
        X_test.append(last_30_days_scaled)
        #Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        #load the model
        model = load_model("repcohome.model")
        #Get the predicted scaled price
        pred_price = model.predict(X_test)
        #undo the scaling 
        pred_price = scaler.inverse_transform(pred_price)
        
        # next day
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price)

        ##visualizations
        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df5['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df5[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df5[['High','Low']])

    elif(choose_stock == "SpiceJet"):

        # get abfrl real time stock price
        df6 = get_history(symbol='spicejet', start=date(2010,1,1), end=date.today())
        df6['Date'] = df6.index

        st.header("SpiceJet NSE Last 5 Days DataFrame:")
        
        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df6.tail())
        
        ## Predictions and adding it to Dashboard
        #Create a new dataframe
        new_df = df6.filter(['Close'])
        #Scale the all of the data to be values between 0 and 1 
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(new_df)
        #Get teh last 30 day closing price 
        last_30_days = new_df[-30:].values
        #Scale the data to be values between 0 and 1
        last_30_days_scaled = scaler.transform(last_30_days)
        #Create an empty list
        X_test = []
        #Append teh past 1 days
        X_test.append(last_30_days_scaled)
        #Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        #load the model
        model = load_model("spicejet.model")
        #Get the predicted scaled price
        pred_price = model.predict(X_test)
        #undo the scaling 
        pred_price = scaler.inverse_transform(pred_price)
        
        # next day
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price)

        ##visualizations
        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df6['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df6[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df6[['High','Low']])

    elif(choose_stock == "TataMotors"):

        # get abfrl real time stock price
        df7 = get_history(symbol='tatamotors', start=date(2010,1,1), end=date.today())
        df7['Date'] = df7.index

        st.header("TataMotors NSE Last 5 Days DataFrame:")
        
        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.dataframe(df7.tail())
        
        ## Predictions and adding it to Dashboard
        #Create a new dataframe
        new_df = df7.filter(['Close'])
        #Scale the all of the data to be values between 0 and 1 
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(new_df)
        #Get teh last 30 day closing price 
        last_30_days = new_df[-30:].values
        #Scale the data to be values between 0 and 1
        last_30_days_scaled = scaler.transform(last_30_days)
        #Create an empty list
        X_test = []
        #Append teh past 1 days
        X_test.append(last_30_days_scaled)
        #Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        #load the model
        model = load_model("tatamotors.model")
        #Get the predicted scaled price
        pred_price = model.predict(X_test)
        #undo the scaling 
        pred_price = scaler.inverse_transform(pred_price)
        
        # next day
        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
        st.markdown(pred_price)

        ##visualizations
        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.area_chart(df7['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.area_chart(df7[['Open','Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df7[['High','Low']])

if __name__ == '__main__':

    main()
