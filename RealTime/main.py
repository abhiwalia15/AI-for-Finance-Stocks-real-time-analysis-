from imports_and_read import *

def main():

    st.title("NSE Real-Time Stocks Analysis")
    st.header("Select the stock and check its next day predicted value")

    # Stock Section
    choose_stock = st.sidebar.selectbox("Choose the Stock ---->>",
        ["NONE","Aditya Birla Fashion Retail Ltd.", "PowerMech Solns."])

    if(choose_stock == "Aditya Birla Fashion Retail Ltd."):

        # get abfrl real time stock price
        df1 = get_history(symbol='abfrl', start=date(2010,1,1), end=date.today())
        df1['Date'] = df.index
        
        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.write(df1.tail())
        

    elif(choose_stock == "PowerMech Solns."):

        # get abfrl real time stock price
        df2 = get_history(symbol='powermech', start=date(2010,1,1), end=date.today())
        df2['Date'] = df.index
        
        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.write(df2.tail())

if __name__ == '__main__':

    main()