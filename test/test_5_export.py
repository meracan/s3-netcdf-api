import os
import numpy as np
import pandas as pd
import json
from netCDF4 import Dataset
from s3netcdf import NetCDF2D
from s3netcdfapi.parameters.parameters import getParameters
from s3netcdfapi.data.get import getData,get
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
  obj=getParameters(netcdf2d,{"variable":"x,y","inode":[0,1]})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['Longitude'].values,[-160.0,-159.899994])
  
  obj=getParameters(netcdf2d,{"variable":"x,y"})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['Longitude'].values,netcdf2d['node','x'])
  np.testing.assert_array_almost_equal(df['Latitude'].values,netcdf2d['node','y'])
  
  # Test 2
  obj=getParameters(netcdf2d,{"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  df2=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_equal(df2['Latitude'].values,[40,40,40,40,40,40])
  np.testing.assert_array_equal(df2['U Velocity,m/s'].values,[0,1,2,10302,10303,10304])
  
  obj=getParameters(netcdf2d,{"variable":"u,v","itime":[0]})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_equal(df['Latitude'].values,netcdf2d['node','y'])
  np.testing.assert_array_equal(df['U Velocity,m/s'].values,np.squeeze(netcdf2d['s','u',0]))
  
  # Test 3
  obj=getParameters(netcdf2d,{"variable":"spectra","isnode":[0],"itime":[0]})
  df3=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df3['Direction,radian'].values,np.tile(np.arange(36)/36.0,(33,1)).flatten())
  np.testing.assert_array_almost_equal(df3['Frequency,Hz'].values,np.tile(np.arange(33)/33.0,(36,1)).T.flatten())
  np.testing.assert_array_equal(df3['VaDens,m2/Hz/degr'].values,np.arange(33*36)/1000000.0)
  
  
  # Test - Datetime
  obj=getParameters(netcdf2d,{"variable":"u","start":"2000-01-01","inode":0})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,np.squeeze(netcdf2d['t','u',0]))
  
  obj=getParameters(netcdf2d,{"variable":"u","start":"2000-01-01T01","inode":0})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,np.squeeze(netcdf2d['t','u',0,1:]))
  
  obj=getParameters(netcdf2d,{"variable":"u","start":"2000-01-01T01","end":"2000-01-01T02","inode":0})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,np.squeeze(netcdf2d['t','u',0,[1,2]]))
  
  obj=getParameters(netcdf2d,{"variable":"u","start":"2000-01-01T01:30","end":"2000-01-01T02:30","inode":0})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,np.squeeze(netcdf2d['t','u',0,[1,2]]))  
  
  obj=getParameters(netcdf2d,{"variable":"u","start":"2000-01-01T01:30","end":"2000-01-01T02:30","inter.temporal":'linear',"inode":0})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[15453,25755])
  
  obj=getParameters(netcdf2d,{"variable":"u","start":"2000-01-01T01:30","end":"2000-01-01T02:30","inter.temporal":'linear',"x":-160.0,"y":40.0})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[15453,25755])
  
  obj=getParameters(netcdf2d,{"variable":"u","start":"2000-01-01T01:30","end":"2000-01-01T02:30","inter.temporal":'linear',"inode":1})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[15454,25756])
  
  obj=getParameters(netcdf2d,{"variable":"u","start":"2000-01-01T01:30","end":"2000-01-01T02:30","inter.temporal":'linear',"inter.mesh":'linear',"x":-159.95,"y":40.0})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[15453.5,25755.5],4)  
  
  # Test - Spatial  
  obj=getParameters(netcdf2d,{"variable":"u","itime":0,"x":-159.95,"y":40.0})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[0.0])
  
  obj=getParameters(netcdf2d,{"variable":"u","itime":0,"x":[-159.95,-159.90],"y":[40.0,40.0]})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[0.0,1.])
  
  obj=getParameters(netcdf2d,{"variable":"u","itime":0,"inter.mesh":'linear',"x":[-159.95,-159.90],"y":[40.0,40.0]})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[0.5,1.],4)
  
  obj=getParameters(netcdf2d,{"variable":"time"})
  df=export.to_table(obj,get(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['Datetime'].values.astype("datetime64[h]").astype("float64"),netcdf2d['time','time'].astype("datetime64[h]").astype("float64"))
  
  
  
  
def test_csv():
  # Test 1
  obj=getParameters(netcdf2d,{"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  export.to_csv(obj,get(netcdf2d,obj))
  
  df=pd.read_csv(obj['filepath']+".csv")
  np.testing.assert_array_equal(df['Longitude'].values,[-160.0,-159.9,-159.8,-160.0,-159.9,-159.8])
  np.testing.assert_array_equal(df['Latitude'].values,[40.0,40.0,40.0,40.0,40.0,40.0])
  np.testing.assert_array_equal(df['U Velocity,m/s'].values,[0.,1.,2.,10302.,10303.,10304.])

def test_json():
  # Test 1
  obj=getParameters(netcdf2d,{"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  export.to_json(obj,get(netcdf2d,obj))
  df=pd.read_json(obj['filepath']+".json")
  np.testing.assert_array_almost_equal(df['Longitude'].values,[-160.0,-159.9,-159.8,-160.0,-159.9,-159.8],5)
  np.testing.assert_array_almost_equal(df['Latitude'].values,[40.0,40.0,40.0,40.0,40.0,40.0],5)
  np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[0.,1.,2.,10302.,10303.,10304.],5)  

def test_geojson():
  # Test 1
  obj=getParameters(netcdf2d,{"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  export.to_geojson(obj,get(netcdf2d,obj))
  with open(obj['filepath']+".geojson") as f:
    geojson = json.load(f)
    data=[feature['properties'] for feature in geojson['features']]
    df=pd.DataFrame(data)
    np.testing.assert_array_almost_equal(df['Longitude'].values,[-160.0,-159.9,-159.8,-160.0,-159.9,-159.8],5)
    np.testing.assert_array_almost_equal(df['Latitude'].values,[40.0,40.0,40.0,40.0,40.0,40.0],5)
    np.testing.assert_array_almost_equal(df['U Velocity,m/s'].values,[0.,1.,2.,10302.,10303.,10304.],5)     

def test_netcdf():
  # Test 1
  obj=getParameters(netcdf2d,{"variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  export.to_netcdf(obj,get(netcdf2d,obj))
  
  with Dataset(obj["filepath"]+".nc", "r") as src_file:
    np.testing.assert_array_equal(src_file.variables['time'][:].astype("datetime64[s]"),np.array(['2000-01-01T00','2000-01-01T01'],dtype="datetime64[h]"))
    np.testing.assert_array_almost_equal(src_file.variables['x'][:],[-160.0,-159.9,-159.8],5)
    np.testing.assert_array_almost_equal(src_file.variables['y'][:],[40.0,40.0,40.0])
    np.testing.assert_array_almost_equal(src_file.variables['u'][:],[[0.,1.,2.],[10302.,10303.,10304.]])  
  

def test_slf():
  obj=getParameters(netcdf2d,{"export":"slf","variable":"u,v","inode":[0,1,2],"itime":[0,1]})
  export.to_slf(obj,get(netcdf2d,obj))


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
  
  test_slf()
  # test_binary()
  # test_mat()
  # test_shapefile()
  # test_tri()