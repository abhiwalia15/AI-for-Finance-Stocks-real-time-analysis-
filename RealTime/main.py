from imports_and_read import * 

def main():

    st.title("NSE Real-Time Stocks Analysis")
    st.header("Select the stock and check its next day predicted value")

    # Stock Section
    choose_stock = st.sidebar.selectbox("Choose the Stock!",
        ["NONE","Aditya Birla Fashion Retail Ltd.", "PowerMech Solns."])

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

        st.subheader("Predictions for the next upcoming day : " + str(NextDay_Date))
        st.markdown(pred_price)

        ##visualizations
        st.subheader("Close Price VS Date Interactive chart for analysis : ")
        
        st.area_chart(df1['Close'])

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

        st.subheader("Predictions for the next upcoming day : " + str(NextDay_Date))
        st.markdown(pred_price)

        ##visualizations
        st.subheader("Close Price VS Date Interactive chart for analysis : ")
        
        st.area_chart(df2['Close'])


if __name__ == '__main__':

    main()