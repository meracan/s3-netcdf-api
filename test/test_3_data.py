import pytest
import numpy as np
from s3netcdf import NetCDF2D
from s3netcdfapi.data.utils import cleanObject,swapAxe,swapAxes
from s3netcdfapi.data.get import get,getData,getDimData


input={
  "name":"input1",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True,
  "maxPartitions":40
}

netcdf2d=NetCDF2D(input)


def test_cleanObject():
  assert cleanObject({"other":['a','b'],"variable":"x","inode":[0],"itime":[1]},['nnode'])=={"variable":"x","inode":[0]}
  assert cleanObject({"other":['a','b'],"variable":"x","inode":[0],"itime":[1]},['nnode','ntime'])=={"variable":"x","inode":[0],"itime":[1]}
  with pytest.raises(Exception):assert cleanObject({"other":['a','b'],"inode":[0],"itime":[1]},['nnode','ntime'])
  

def test_swapAxe():
  data=np.zeros(2*3*4*5).reshape((2,3,4,5))
  dimensionNames=["A","B","C","D"]
  data,dimensionNames=swapAxe(data,dimensionNames,"B")
  assert data.shape==(3,2,4,5)
  assert dimensionNames==["B","A","C","D"]
  data,dimensionNames=swapAxe(data,dimensionNames,"C")
  assert data.shape==(4,2,3,5)  
  assert dimensionNames==["C","A","B","D"]
  
  
def test_swapAxes():
  data=np.zeros(2*3*4*5).reshape((2,3,4,5))
  dimensionNames=["A","B","C","D"]
  data,dimensionNames=swapAxes(data,dimensionNames,["D","A","B","C"])
  assert data.shape==(5,2,3,4)
  assert dimensionNames==["D","A","B","C"]
  
def test_getDimData():
  # Test 1
  obj={"user_sxy":False,"user_xy":False,"user_time":False,"dataOnly":False}
  dimensions=['ntime','nnode']
  variables=getDimData(netcdf2d,obj,dimensions)
  np.testing.assert_array_equal(variables['time']['data'],netcdf2d['time','time'])
  assert variables['node']['data']==None
  np.testing.assert_array_equal(variables['node']['subdata']['x']['data'],netcdf2d['node','x'])
  np.testing.assert_array_equal(variables['node']['subdata']['y']['data'],netcdf2d['node','y'])
  
  # Test 2
  dimensions=['ntime','nsnode']
  variables=getDimData(netcdf2d,obj,dimensions)
  np.testing.assert_array_equal(variables['snode']['subdata']['sx']['data'],netcdf2d['snode','sx'])
  np.testing.assert_array_equal(variables['snode']['subdata']['sy']['data'],netcdf2d['snode','sy'])
  

def test_getData():
  # Test 1  
  obj={"user_sxy":False,"user_xy":False,"user_time":False,"dataOnly":False}
  np.testing.assert_array_equal(getData(netcdf2d,{**obj,"inode":[0,1]},"x")['data'],netcdf2d['node','x',[0,1]])
  np.testing.assert_array_equal(getData(netcdf2d,{**obj,"itime":[0]},"u",["ntime","nnode"])['data'],netcdf2d['s','u',0])
  np.testing.assert_array_equal(getData(netcdf2d,{**obj,"isnode":[0],"itime":[0]},"spectra",["nsnode","ntime","nfreq","ndir"])['data'],netcdf2d['spc','spectra',0,0])

if __name__ == "__main__":
  test_cleanObject()
  test_swapAxe()
  test_swapAxes()
  
  test_getDimData()
  test_getData()