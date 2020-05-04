from imports_and_read import *

symbol = input("Enter symbol of stock\n")
st.write(symbol)

df = get_history(symbol=symbol, start=date(2010,1,1), end=date.today())
df['Date'] = df.index




