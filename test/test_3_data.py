import pytest
import numpy as np
from s3netcdfapi import S3NetCDFAPI
from s3netcdfapi.data.utils import cleanObject,swapAxe,swapAxes
from s3netcdfapi.data.getData import getData,_getData,getDimData


input={
  "name":"input1",
  "cacheLocation":"../s3",
  "apiCacheLocation":"../s3/tmp",
  "localOnly":True,
  "verbose":True,
  "maxPartitions":40
}

netcdf2d=S3NetCDFAPI(input)


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
  obj=netcdf2d.prepareInput({'variable':'u'})
  variables=getDimData(netcdf2d,obj,['ntime','nnode'])
  np.testing.assert_array_equal(variables['time']['data'],netcdf2d['time','time'])
  assert variables['node']['data']==None
  np.testing.assert_array_equal(variables['node']['subdata']['x']['data'],netcdf2d['node','x'])
  np.testing.assert_array_equal(variables['node']['subdata']['y']['data'],netcdf2d['node','y'])
  
  # Test 2
  obj=netcdf2d.prepareInput({'variable':'spectra'})
  variables=getDimData(netcdf2d,obj,['ntime','nsnode','nfreq','ndir'])
  np.testing.assert_array_equal(variables['snode']['subdata']['x']['data'],netcdf2d['snode','sx'])
  np.testing.assert_array_equal(variables['snode']['subdata']['y']['data'],netcdf2d['snode','sy'])
  

def test_getData():
  # Test 1
  # Indices
  np.testing.assert_array_equal(_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0,1],"variable":"x"}),"x")['data'],netcdf2d['node','x',[0,1]])
  np.testing.assert_array_equal(_getData(netcdf2d,netcdf2d.prepareInput({"itime":[0],"variable":"u"}),"u",["ntime","nnode"])['data'],netcdf2d['s','u',0])
  np.testing.assert_array_equal(_getData(netcdf2d,netcdf2d.prepareInput({"isnode":[0],"itime":[0],"variable":"spectra"}),"spectra",["nsnode","ntime","nfreq","ndir"])['data'],netcdf2d['spc','spectra',0,0])

def test_getData_SWAN():
  input={
    "name":"SWANv5",
    "bucket":"uvic-bcwave",
    "cacheLocation":"../s3",
    "apiCacheLocation":"../s3/tmp",
    "localOnly":False,
    "verbose":True,
    "maxPartitions":10
  }
  netcdf2d=S3NetCDFAPI(input)
  
  # np.testing.assert_array_equal(_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0,1],"variable":"lon"}),"lon")['data'],netcdf2d['nodes','lon',[0,1]])
  # np.testing.assert_array_equal(_getData(netcdf2d,netcdf2d.prepareInput({"variable":"lon"}),"lon")['data'],netcdf2d['nodes','lon'])
  # np.testing.assert_array_equal(_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0,1],"variable":"lat"}),"lat")['data'],netcdf2d['nodes','lat',[0,1]])
  # np.testing.assert_array_equal(_getData(netcdf2d,netcdf2d.prepareInput({"variable":"lat"}),"lat")['data'],netcdf2d['nodes','lat'])  
  
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0,1],"variable":"lon"}),"lon")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0,1],"variable":"lat"}),"lat")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0],"variable":"hs"}),"hs")['data'])
  
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"lon":list(netcdf2d['nodes','lon',0]),"lat":list(netcdf2d['nodes','lat',0]),"variable":"hs"}),"hs")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"lon":list(netcdf2d['nodes','lon',0:10]),"lat":list(netcdf2d['nodes','lat',0:10]),"variable":"hs"}),"hs")['data'])
  
  
  # Interpolation
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"inter.mesh":"linear","lon":list(netcdf2d['nodes','lon',0:70]),"lat":list(netcdf2d['nodes','lat',0:70]),"variable":"hs"}),"hs")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"inter.mesh":"linear","lon":[-170,-130],"lat":[35,52],"itime":[0,1],"variable":"hs"}),"hs")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"inter.mesh":"linear","lon":[-170,-130],"lat":[35,52],"variable":"hs"}),"hs")['data'])
  
  # Outside point
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"extra.mesh":"none","inter.mesh":"linear","lon":[-170,-129,-130],"lat":[35,53,52],"itime":[0,1,2,3,4],"variable":"hs"}),"hs")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"extra.mesh":"nearest","inter.mesh":"linear","lon":[-170,-129,-130],"lat":[35,53,52],"itime":[0,1,2,3,4],"variable":"hs"}),"hs")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"itime":[0,1,2,3,4],"inode":39,"variable":"hs"}),"hs")['data'])
  
  

def test_getDataLimits():
  print(_getData(netcdf2d,netcdf2d.prepareInput({"x":0.0,"y":0.0,"variable":"u","itime":0,"inter.mesh":"linear"}),"u")['data'])

if __name__ == "__main__":
  # test_cleanObject()
  # test_swapAxe()
  # test_swapAxes()
  
  # test_getDimData()
  # test_getData()
  test_getData_SWAN()
  # test_getDataLimits()