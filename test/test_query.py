import pytest
import numpy as np
from s3netcdf import NetCDF2D
from s3netcdfapi.query.utils import cleanObject,swapAxe,swapAxes
from s3netcdfapi.query.get import getData,getHeader,getDimensionValues


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
  
    
def test_getHeader():
  assert getHeader(netcdf2d,"x")=="Longitude"
  assert getHeader(netcdf2d,"u")=="U Velocity,m/s"
  
    
def test_getDimensionValues():
  # Test 1
  shape=(2,2)
  obj={"x":[0.0,1.0],"y":[0.0,1.0],"datetime":np.array(['2000-01-01','2000-01-02'],dtype="datetime64[h]")}
  dimensions=['ntime','nnode']
  np.testing.assert_array_equal(getDimensionValues(netcdf2d,shape,obj,dimensions)[0],
  np.array([['2000-01-01T00Z,0.0,0.0', '2000-01-01T00Z,1.0,1.0'],['2000-01-02T00Z,0.0,0.0', '2000-01-02T00Z,1.0,1.0']]))
  
  # Test 2
  shape=(2,2,2,2)
  obj={"x":[0.0,1.0],"y":[0.0,1.0],"datetime":np.array(['2000-01-01','2000-01-02'],dtype="datetime64[h]"),"freq":[0.1,0.2],"dir":[1,2]}
  dimensions=['nnode','ntime','nfreq','ndir']
  
  np.testing.assert_array_equal(getDimensionValues(netcdf2d,shape,obj,dimensions)[0],np.array(
    [[[['0.0,0.0,2000-01-01T00Z,0.1,1','0.0,0.0,2000-01-01T00Z,0.1,2'],
   ['0.0,0.0,2000-01-01T00Z,0.2,1','0.0,0.0,2000-01-01T00Z,0.2,2']],

  [['0.0,0.0,2000-01-02T00Z,0.1,1','0.0,0.0,2000-01-02T00Z,0.1,2'],
   ['0.0,0.0,2000-01-02T00Z,0.2,1','0.0,0.0,2000-01-02T00Z,0.2,2']]],


 [[['1.0,1.0,2000-01-01T00Z,0.1,1','1.0,1.0,2000-01-01T00Z,0.1,2'],
   ['1.0,1.0,2000-01-01T00Z,0.2,1','1.0,1.0,2000-01-01T00Z,0.2,2']],

  [['1.0,1.0,2000-01-02T00Z,0.1,1','1.0,1.0,2000-01-02T00Z,0.1,2'],
   ['1.0,1.0,2000-01-02T00Z,0.2,1','1.0,1.0,2000-01-02T00Z,0.2,2']]]]
    ))
  
  

def test_getData():
  
  # np.testing.assert_array_equal(getData(netcdf2d,{"inode":[0,1]},"x")['data'],netcdf2d['node','x',[0,1]])
  # np.testing.assert_array_equal(getData(netcdf2d,{"itime":[0]},"u",["ntime","nnode"])['data'],netcdf2d['s','u',0])
  # np.testing.assert_array_equal(getData(netcdf2d,{"isnode":[0],"itime":[0]},"spectra",["nsnode","ntime","nfreq","ndir"])['data'],netcdf2d['spc','spectra',0,0])
  # print(getData(netcdf2d,{"inode":[0,1]},"x")['data'])
  # print(getData(netcdf2d,{"inode":[0,1]},"y")['data'])

if __name__ == "__main__":
  # test_cleanObject()
  # test_swapAxe()
  # test_swapAxes()
  # test_getHeader()
  # test_getDimensionValues()
  test_getData()