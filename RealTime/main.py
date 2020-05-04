from imports_and_read import *

df = get_history(symbol="abfrl", start=date(2010,1,1), end=date.today())
df['Date'] = df.index

print(df.tail())

