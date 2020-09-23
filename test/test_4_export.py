import numpy as np
import pandas as pd
import json
from netCDF4 import Dataset
from s3netcdf import NetCDF2D
from s3netcdfapi.parameters.parameters import getParameters
from s3netcdfapi.query.get import getData,getAllData
import s3netcdfapi.export as export
input={
  "name":"input1",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True,
  "maxPartitions":40
}

netcdf2d=NetCDF2D(input)


def test_table():
  # Test 1
  obj1=getParameters(netcdf2d,{"variable":"x,y","inode":[0,1]})
  df1=export.to_table(obj1,getAllData(netcdf2d,obj1))
  np.testing.assert_array_almost_equal(df1['Longitude'].values,[-160.0,-159.899994])
  
  # Test 2
  obj2=getParameters(netcdf2d,{"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  df2=export.to_table(obj2,getAllData(netcdf2d,obj2))
  np.testing.assert_array_equal(df2['Latitude'].values,[40,40,40,40,40,40])
  np.testing.assert_array_equal(df2['U Velocity,m/s'].values,[0,1,2,10302,10303,10304])
  
  # Test 3
  obj3=getParameters(netcdf2d,{"variable":"spectra","isnode":[0],"itime":[0]})
  df3=export.to_table(obj3,getAllData(netcdf2d,obj3))
  np.testing.assert_array_almost_equal(df3['Direction,radian'].values,np.tile(np.arange(36)/36.0,(33,1)).flatten())
  np.testing.assert_array_almost_equal(df3['Frequency,Hz'].values,np.tile(np.arange(33)/33.0,(36,1)).T.flatten())
  np.testing.assert_array_equal(df3['VaDens,m2/Hz/degr'].values,np.arange(33*36)/1000000.0)
  
def test_csv():
  # Test 1
  obj=getParameters(netcdf2d,{"dataOnly":False,"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  export.to_csv(obj,getAllData(netcdf2d,obj))
  
  df=pd.read_csv(obj['output']+".csv")
  np.testing.assert_array_equal(df['Longitude'].values,[-160.0,-159.9,-159.8,-160.0,-159.9,-159.8])
  np.testing.assert_array_equal(df['Latitude'].values,[40.0,40.0,40.0,40.0,40.0,40.0])
  np.testing.assert_array_equal(df['U Velocity,m/s'].values,[0.,1.,2.,10302.,10303.,10304.])

def test_json():
  # Test 1
  obj=getParameters(netcdf2d,{"dataOnly":False,"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  export.to_json(obj,getAllData(netcdf2d,obj))
  df=pd.read_json(obj['output']+".json")
  np.testing.assert_array_almost_equal(df['Longitude'].values,[-160.0,-159.9,-159.8,-160.0,-159.9,-159.8],5)
  np.testing.assert_array_almost_equal(df['Latitude'].values,[40.0,40.0,40.0,40.0,40.0,40.0],5)
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[0.,1.,2.,10302.,10303.,10304.],5)  

def test_geojson():
  # Test 1
  obj=getParameters(netcdf2d,{"dataOnly":False,"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  export.to_geojson(obj,getAllData(netcdf2d,obj))
  with open(obj['output']+".geojson") as f:
    geojson = json.load(f)
    data=[feature['properties'] for feature in geojson['features']]
    df=pd.DataFrame(data)
    np.testing.assert_array_almost_equal(df['Longitude'].values,[-160.0,-159.9,-159.8,-160.0,-159.9,-159.8],5)
    np.testing.assert_array_almost_equal(df['Latitude'].values,[40.0,40.0,40.0,40.0,40.0,40.0],5)
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[0.,1.,2.,10302.,10303.,10304.],5)     

def test_netcdf():
  # Test 1
  obj=getParameters(netcdf2d,{"dataOnly":False,"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  export.to_netcdf(obj,getAllData(netcdf2d,obj))
  
  with Dataset(obj["output"]+".nc", "r") as src_file:
    np.testing.assert_array_equal(src_file.variables['time'][:].astype("datetime64[s]"),np.array(['2000-01-01T00','2000-01-01T01'],dtype="datetime64[h]"))
    np.testing.assert_array_almost_equal(src_file.variables['x'][:],[-160.0,-159.9,-159.8],5)
    np.testing.assert_array_almost_equal(src_file.variables['y'][:],[40.0,40.0,40.0])
    np.testing.assert_array_almost_equal(src_file.variables['u'][:],[[0.,1.,2.],[10302.,10303.,10304.]])  
  

def test_slf():
  obj=getParameters(netcdf2d,{"export":"slf","dataOnly":True,"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  export.to_slf(obj,getAllData(netcdf2d,obj))


# def test_binary():
#     export.binary()
    

# def test_mat():
#     export.mat()


# def test_shapefile():
#     export.shapefile()


# def test_tri():
#     export.tri()

if __name__ == "__main__":
  test_table()
  test_csv()
  test_json()
  test_geojson()
  test_netcdf()
  
  # test_slf()
  # test_binary()
  # test_mat()
  # test_shapefile()
  # test_tri()