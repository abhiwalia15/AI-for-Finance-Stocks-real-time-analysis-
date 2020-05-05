from imports_and_read import *

st.title("NSE Real-Time Stocks Analysis")
st.header("Enter the symbol of the stock you want to analyse and we will predict the next day Close Price for you")

#symbol = input("Enter symbol of stock\n")
symbol = st.text_input("Enter symbol")

df = get_history(symbol=symbol, start=date(2010,1,1), end=date.today())
df['Date'] = df.index

st.dataframe(df)


