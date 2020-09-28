import os
import numpy as np
import pandas as pd
import json
from netCDF4 import Dataset
from s3netcdfapi import S3NetCDFAPI
from s3netcdfapi.data import getData
import s3netcdfapi.export as export
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
  # Test 1
  obj=netcdf2d.prepareInput({"variable":"lon,lat","inode":[0,1]})
  r=getData(netcdf2d,obj)
  np.testing.assert_array_almost_equal(r['lon']['data'],[0,0.1])
  np.testing.assert_array_almost_equal(r['lat']['data'],[0,0])
  
  obj=netcdf2d.prepareInput({"variable":"lon,lat"})
  r=getData(netcdf2d,obj)
  np.testing.assert_array_almost_equal(r['lon']['data'],netcdf2d['node','lon'])
  np.testing.assert_array_almost_equal(r['lat']['data'],netcdf2d['node','lat'])
 
  # Test 2
  obj=netcdf2d.prepareInput({"variable":"hs,u10","inode":[0,1,2],"itime":[0,1]})
  r=getData(netcdf2d,obj)
  np.testing.assert_array_almost_equal(r['hs']['data'],[[0,1,2],[10201,10202,10203]])
  
  obj=netcdf2d.prepareInput({"variable":"spectra","x":0,"y":0,"itime":[0]})
  r=getData(netcdf2d,obj)

if __name__ == "__main__":
  test_table()