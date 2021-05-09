import pandas as pd    
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from urllib.parse import urlencode, unquote

pd.plotting.register_matplotlib_converters()

address = "https://api.meracan.ca/?"
query = urlencode({
  "export":"csv", 
  "variable":"lon,lat,bed", 
  "inode":":10000"
})


df = pd.read_csv(address+query)
lons = df["longitude,degrees_east"]
lats = df["latitude,degrees_north"]
bed = df["Bathymetry,m"]

ax = plt.axes(projection='3d')
plt.title("ocean bed")
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_zlabel('bathymetry (metres)')
ax.scatter(lons,lats,bed, marker=".", s=1)
plt.grid(True)
plt.show()  