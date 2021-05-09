import os
import numpy as np
import pandas as pd
import json
from netCDF4 import Dataset
from s3netcdfapi import S3NetCDFAPI
from s3netcdfapi.data import getData
import s3netcdfapi.export as export
import binpy
input={
  "name":"input2",
  "cacheLocation":"../s3",
  "apiCacheLocation":"../s3/tmp",
  "localOnly":True,
  "verbose":True,
  "maxPartitions":40
}

netcdf2d=S3NetCDFAPI(input)


def test_table():
  # # Test Lon/Lat
  # obj=netcdf2d.prepareInput({"variable":"lon,lat","inode":[0,1]})
  # r=getData(netcdf2d,obj)
  # np.testing.assert_array_almost_equal(r['lon']['data'],[0,0.1])
  # np.testing.assert_array_almost_equal(r['lat']['data'],[0,0])
  
  # obj=netcdf2d.prepareInput({"variable":"lon,lat"})
  # r=getData(netcdf2d,obj)
  # np.testing.assert_array_almost_equal(r['lon']['data'],netcdf2d['node','lon'])
  # np.testing.assert_array_almost_equal(r['lat']['data'],netcdf2d['node','lat'])
 
  # Test Elem
  obj=netcdf2d.prepareInput({"variable":"elem"})
  df=export.to_table(obj,getData(netcdf2d,obj))
  np.testing.assert_array_almost_equal(df['n1'],netcdf2d['elem','elem'][:,0])
  np.testing.assert_array_almost_equal(df['n2'],netcdf2d['elem','elem'][:,1])
  np.testing.assert_array_almost_equal(df['n3'],netcdf2d['elem','elem'][:,2])
 
 
  # # Test 2
  # obj=netcdf2d.prepareInput({"variable":"hs,u10","inode":[0,1,2],"itime":[0,1]})
  # r=getData(netcdf2d,obj)
  # np.testing.assert_array_almost_equal(r['hs']['data'],[[0,1,2],[10201,10202,10203]])
  
  # obj=netcdf2d.prepareInput({"variable":"spectra","x":0,"y":0,"itime":[0]})
  # r=getData(netcdf2d,obj)

def test_binary():
  obj=netcdf2d.prepareInput({"export":"bin","variable":"mesh"})
  export.to_binary(netcdf2d,obj,getData(netcdf2d,obj))
  with open(obj["filepath"]+".bin","rb") as f:
    results=binpy.read(f)
  id=netcdf2d.name
  np.testing.assert_array_almost_equal(results[id+'_elem'],netcdf2d['elem','elem'])
  np.testing.assert_array_almost_equal(results[id+'_x'],netcdf2d['node','lon'])
  np.testing.assert_array_almost_equal(results[id+'_y'],netcdf2d['node','lat'])
  
  obj=netcdf2d.prepareInput({"export":"bin","variable":"u10,v10","itime":"1:3"})
  export.to_binary(netcdf2d,obj,getData(netcdf2d,obj))
  with open(obj["filepath"]+".bin","rb") as f:
    results=binpy.read(f)
  id=netcdf2d.name
  np.testing.assert_array_almost_equal(results[id+'_u10_ntime_1'],np.squeeze(netcdf2d['s','u10',1]))
  np.testing.assert_array_almost_equal(results[id+'_u10_ntime_2'],np.squeeze(netcdf2d['s','u10',2]))
  np.testing.assert_array_almost_equal(results[id+'_v10_ntime_1'],np.squeeze(netcdf2d['s','v10',1]))
  np.testing.assert_array_almost_equal(results[id+'_v10_ntime_2'],np.squeeze(netcdf2d['s','v10',2]))  

  obj=netcdf2d.prepareInput({"export":"bin","variable":"time"})
  export.to_binary(netcdf2d,obj,getData(netcdf2d,obj))
  with open(obj["filepath"]+".bin","rb") as f:
    results=binpy.read(f)
  id=netcdf2d.name
  np.testing.assert_array_equal(results[id+'_time'],netcdf2d['time','time'])
  

if __name__ == "__main__":
  test_table()
  # test_binary()
  