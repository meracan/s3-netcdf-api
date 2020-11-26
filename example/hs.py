import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlencode, unquote

pd.plotting.register_matplotlib_converters() # avoids warning

address = "https://api.meracan.ca/?"
query = urlencode({
  "export":"csv", 
  "variable":"hs", 
  "inode":"0", 
  "start":"2004-07-01", 
  "end":"2004-08-01"
})

df = pd.read_csv(address+query)
t = df["Datetime"].astype("datetime64[ns]")
y = df["hs,m"]

plt.plot(t,y)
plt.title(f"significant wave height of node 0, in metres")
plt.xlabel("datetime")
plt.ylabel(f"hs (m)")
plt.grid(True)
plt.xticks(rotation=30)
plt.show()
