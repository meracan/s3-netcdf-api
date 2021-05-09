import pytest
import numpy as np
from s3netcdfapi import S3NetCDFAPI
from s3netcdfapi.data.utils import cleanObject,swapAxe,swapAxes
from s3netcdfapi.data.getData import getData,_getData,getDimData


input={
  "name":"s3netcdfapi_test",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True,
  "maxPartitions":40,
  
}

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
  with S3NetCDFAPI(input) as netcdf:
    # Test 1
    obj=netcdf.prepareInput({'variable':'u'})
    variables=getDimData(netcdf,obj,['ntime','nnode'])
    np.testing.assert_array_equal(variables['time']['data'],netcdf['time','time'])
    assert variables['node']['data']==None
    np.testing.assert_array_equal(variables['node']['subdata']['x']['data'],netcdf['node','x'])
    np.testing.assert_array_equal(variables['node']['subdata']['y']['data'],netcdf['node','y'])
    
    # Test 2
    obj=netcdf.prepareInput({'variable':'spectra'})
    variables=getDimData(netcdf,obj,['ntime','nsnode','nfreq','ndir'])
    np.testing.assert_array_equal(variables['snode']['subdata']['x']['data'],netcdf['snode','sx'])
    np.testing.assert_array_equal(variables['snode']['subdata']['y']['data'],netcdf['snode','sy'])
  

def test_getData():
  with S3NetCDFAPI(input) as netcdf:
    # Test X,Y
    # Indices
    np.testing.assert_array_equal(_getData(netcdf,netcdf.prepareInput({"inode":[0,1],"variable":"x"}),"x")['data'],netcdf['node','x',[0,1]])
    np.testing.assert_array_equal(_getData(netcdf,netcdf.prepareInput({"itime":[0],"variable":"u"}),"u",["ntime","nnode"])['data'],netcdf['s','u',0])
    np.testing.assert_array_equal(_getData(netcdf,netcdf.prepareInput({"isnode":[0],"itime":[0],"variable":"spectra"}),"spectra",["nsnode","ntime","nfreq","ndir"])['data'],netcdf['spc','spectra',0,0])
    np.testing.assert_array_equal(_getData(netcdf,netcdf.prepareInput({"variable":"elem"}),"elem")['data'],netcdf['elem','elem'])
    
    # Test Station Name and Ids
    np.testing.assert_array_equal(getData(netcdf,netcdf.prepareInput({"variable":"name"}))['name']['data'],np.array(['a','b','c','d','e','f']))
    np.testing.assert_array_equal(getData(netcdf,netcdf.prepareInput({"variable":"stationid"}))['stationid']['data'],netcdf['snode','stationid'])  
    
    # Test Spectra
    np.testing.assert_array_equal(getData(netcdf,netcdf.prepareInput({"variable":"spectra","itime":0}))['spectra']['data'].shape,(10,1,33,36))
    np.testing.assert_array_equal(getData(netcdf,netcdf.prepareInput({"variable":"spectra","itime":0,"ifreq":5}))['spectra']['data'].shape,(10,1,1,36))
    np.testing.assert_array_equal(getData(netcdf,netcdf.prepareInput({"variable":"spectra","itime":0,"ifreq":5,"idir":0}))['spectra']['data'].shape,(10,1,1,1))
    np.testing.assert_array_equal(getData(netcdf,netcdf.prepareInput({"variable":"spectra","itime":0,"ifreq":5,"idir":0}))['spectra']['data'].shape,(10,1,1,1))
    np.testing.assert_array_equal(getData(netcdf,netcdf.prepareInput({"variable":"spectra","itime":0,"ifreq":5,"idir":0,"x":-160,"y":40.0}))['spectra']['data'].shape,(1,1,1,1))
    np.testing.assert_array_equal(getData(netcdf,netcdf.prepareInput({"variable":"spectra","station":"f"}))['spectra']['data'].shape,(3,1000,33,36))
    np.testing.assert_array_equal(getData(netcdf,netcdf.prepareInput({"variable":"spectra","isnode":0}))['spectra']['data'],netcdf['spc','spectra',0])
    np.testing.assert_array_equal(getData(netcdf,netcdf.prepareInput({"variable":"spectra","isnode":0,"start":"2000-01-01T00:00","end":"2000-01-02T00:00"}))['spectra']['data'].shape,(1,25,33,36))
    np.testing.assert_array_equal(getData(netcdf,netcdf.prepareInput({"variable":"spectra","start":"2000-01-01T00:00","end":"2000-01-02T00:00","x":-150.0,"y":50.0}))['spectra']['dimData']['snode']['subdata']['stationname']['data'],np.array(['f']))
    np.testing.assert_array_equal(getData(netcdf,netcdf.prepareInput({"variable":"spectra","start":"2000-01-01T00:00","end":"2000-01-01T00:00","x":-150.0,"y":50.0}))['spectra']['dimData']['snode']['subdata']['stationname']['data'],np.array(['f']))
    

def test_getDataLimits():
  with S3NetCDFAPI(input) as netcdf:
    print(_getData(netcdf,netcdf.prepareInput({"x":0.0,"y":0.0,"variable":"u","itime":0,"inter.mesh":"linear"}),"u")['data'])

if __name__ == "__main__":
  test_cleanObject()
  test_swapAxe()
  test_swapAxes()
  
  test_getDimData()
  test_getData()
  
  # test_getDataLimits() #TODO