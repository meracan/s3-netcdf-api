import pandas as pd




def matplotlib_example():
    import matplotlib.pyplot as plt
    df = pd.read_csv("https://w4wj62194j.execute-api.us-west-2.amazonaws.com/Prod?variable=hs&inode=0&export=csv")
    plt.plot_date(df['Datetime'].astype("datetime64[h]"),df['hs,m'],"-")
    plt.savefig("hs_0.matplotlib.png")

def matplotlib_spectra_example():
    import matplotlib.pyplot as plt
    from netCDF4 import Dataset,chartostring
    import requests
    import numpy as np
    
    # Download data
    start='2008-12-06T01'
    end='2008-12-07T01'
    x=-125.55
    y=48.92
    url="https://api.meracan.ca?variable=spectra&export=netcdf&start={}&end={}&x={}&y={}".format(start,end,x,y)
    # Development path
    # url="https://w4wj62194j.execute-api.us-west-2.amazonaws.com/Prod?variable=spectra&export=netcdf&start={}&end={}&x={}&y={}".format(start,end,x,y)
    r = requests.get(url, allow_redirects=True)
    open('tmp.nc', 'wb').write(r.content)
    
    # dataset = Dataset(url)
    nc = Dataset('tmp.nc')
    time=nc.variables['time'][:].astype("datetime64[s]")
    x=nc.variables['x'][:]
    y=nc.variables['y'][:]
    freq=nc.variables['freq'][:]
    dir=nc.variables['dir'][:]
    spectra=nc.variables['spectra'][:] # shape(node,time,freq,dir)
    stationName=chartostring(nc.variables['stationname'][:].astype("S1"))
    stationId=nc.variables['stationid'][:]
    
    
    # Using linspace so that the endpoint of 360 is included
    dirRad = np.radians(dir)
 
     
    r, theta = np.meshgrid(freq, dirRad)
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    for inode in range(spectra.shape[0]):
      for itime in range(spectra.shape[1]):
        values =spectra[inode,itime].T # Transpose to make it work with matplotlib
        ax.contourf(theta, r, values)
        plt.savefig("spectra.{}_{}.png".format(stationName[inode],time[itime]))


def plotly_example():
    import plotly.graph_objects as go
    df = pd.read_csv("https://w4wj62194j.execute-api.us-west-2.amazonaws.com/Prod?variable=hs&inode=0&export=csv")
    fig = go.Figure([go.Scatter(x=df['Datetime'], y=df['hs,m'])])
    fig.write_image("hs_0.plotly.png")


    
    
if __name__ == "__main__":
  # matplotlib_example()
  # plotly_example()
  matplotlib_spectra_example()
      