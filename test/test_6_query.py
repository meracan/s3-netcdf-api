from s3netcdfapi import S3NetCDFAPI
import pandas as pd

def test_query_SWANv5():
    
    # print(netcdf2d.run({"id":"input1","export":"csv","variable":"u,v","x":-160.0,"y":40.0,"itime":[0,1]}))
    # print(netcdf2d.run({}))
    
    netcdf2d=S3NetCDFAPI.init({"id":"SWANv5","bucket":"uvic-bcwave","localOnly":False},{})
    
    # Elem
    # df = pd.read_csv("https://api.meracan.ca?variable=elem&export=csv")
    # print(netcdf2d.run({"export":"csv","variable":"spectra","station":"beverly","itime":[0,1]}))
    # 1 variable
    # print(netcdf2d.run({"export":"csv","variable":"hs","inode":0,"itime":[0,1]}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","inode":[0,1],"itime":[0,1]}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","itime":0}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","inode":0}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","x":-136.7264224,"y":57.39504017,"itime":[0,1,2,3,4,5]}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","x":[-136.7264224,-135.0],"y":[57.39504017,57.],"itime":[0,1,2,3,4,5]}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","inode":0,"start":"2014-01-01","end":"2014-01-10"}))
    
    # Error testing
    # print(netcdf2d.run({"export":"csv","variable":"hs","inode":0,"start":"2003-01-01","end":"2019-01-10"}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","inode":0,"start":"2016-01-01","end":"2019-01-10"}))
    # print(netcdf2d.run({"export":"csv","variable":"hs","x":-160.0,"y":0,"itime":0,"inter.mesh":"linear"}))
    
    
    # print(netcdf2d.run({"export":"csv","variable":"spectra","x":-125.55,"y":48.92,"start":"2010-02-02T02","end":"2010-02-02T02"}))
    print(netcdf2d.run({"variable":"time"}))
    
    # print(netcdf2d.run({"export":"bin","variable":"elem"}))
    # print(netcdf2d.run({}))

def test_query_POLAR():
    netcdf2d=S3NetCDFAPI.init({"id":"netcdf","bucket":"meracan-polar","localOnly":False},{})
    # print(netcdf2d.run({}))
    print(netcdf2d.run({"export":"csv","variable":"fs","inode":271608}))
    
    
if __name__ == "__main__":
  # test_query_SWANv5()
  test_query_POLAR()
  