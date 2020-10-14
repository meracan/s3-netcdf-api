import pytest
import numpy as np
from s3netcdfapi import S3NetCDFAPI
from s3netcdfapi.data.utils import cleanObject,swapAxe,swapAxes
from s3netcdfapi.data.getData import getData,_getData,getDimData

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
  
  np.testing.assert_array_equal(_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0,1],"variable":"lon"}),"lon")['data'],netcdf2d['nodes','lon',[0,1]])
  np.testing.assert_array_equal(_getData(netcdf2d,netcdf2d.prepareInput({"variable":"lon"}),"lon")['data'],netcdf2d['nodes','lon'])
  np.testing.assert_array_equal(_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0,1],"variable":"lat"}),"lat")['data'],netcdf2d['nodes','lat',[0,1]])
  np.testing.assert_array_equal(_getData(netcdf2d,netcdf2d.prepareInput({"variable":"lat"}),"lat")['data'],netcdf2d['nodes','lat'])  
  
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0,1],"variable":"lon"}),"lon")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0,1],"variable":"lat"}),"lat")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0],"variable":"hs"}),"hs")['data'])
  
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"lon":list(netcdf2d['nodes','lon',0]),"lat":list(netcdf2d['nodes','lat',0]),"variable":"hs"}),"hs")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"lon":list(netcdf2d['nodes','lon',0:10]),"lat":list(netcdf2d['nodes','lat',0:10]),"variable":"hs"}),"hs")['data'])
  
  
  # # Interpolation
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"inter.mesh":"linear","lon":list(netcdf2d['nodes','lon',0:70]),"lat":list(netcdf2d['nodes','lat',0:70]),"variable":"hs"}),"hs")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"inter.mesh":"linear","lon":[-170,-130],"lat":[35,52],"itime":[0,1],"variable":"hs"}),"hs")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"inter.mesh":"linear","lon":[-170,-130],"lat":[35,52],"variable":"hs"}),"hs")['data'])
  
  # # Outside point
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"extra.mesh":"none","inter.mesh":"linear","lon":[-170,-129,-130],"lat":[35,53,52],"itime":[0,1,2,3,4],"variable":"hs"}),"hs")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"extra.mesh":"nearest","inter.mesh":"linear","lon":[-170,-129,-130],"lat":[35,53,52],"itime":[0,1,2,3,4],"variable":"hs"}),"hs")['data'])
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"itime":[0,1,2,3,4],"inode":39,"variable":"hs"}),"hs")['data'])
  
  
  # print(_getData(netcdf2d,netcdf2d.prepareInput({"itime":0,"variable":"hs"}),"hs")['data'])
  


if __name__ == "__main__":
  test_getData_SWAN()
  