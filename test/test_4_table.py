import numpy as np
import pandas as pd
import json

from s3netcdfapi import S3NetCDFAPI
from s3netcdfapi.data.getData import _getData
from s3netcdfapi.export.table import _getMeta,getMeta,combineValues,dimData2Table
import s3netcdfapi.export as export

input={
  "name":"s3netcdfapi_test",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True,
  "maxPartitions":40
}



def test__getMeta():
  data={
    "name":"dummy",
    "meta":{"standard_name":"Name","units":"m"}
  }
  
  assert _getMeta(data,type="header")=="Name,m"
  assert _getMeta(data,type="units")=="m"
 
def test_getMeta():
  with S3NetCDFAPI(input) as netcdf:
    data=_getData(netcdf,netcdf.prepareInput({"inode":[0,1],"variable":"u"}),"u")
    dimData=data['dimData']
   
    assert getMeta(dimData,"header")==['Longitude','Latitude','Datetime']
    assert getMeta(dimData,"type")==['f','f','d']
    assert getMeta(dimData,'header',data,True)['x']=="Longitude"
    assert getMeta(dimData,'header',data,True)['y']=="Latitude"
    assert getMeta(dimData,'header',data,True)['time']=="Datetime"


def test_combineValues():
  with S3NetCDFAPI(input) as netcdf:
    data=_getData(netcdf,netcdf.prepareInput({"inode":[0,1],"variable":"u"}),"u")
    dimData=data['dimData']
    assert combineValues(dimData['node']['subdata'])[0]=='-160.0,40.0'
   
 
def test_dimData2Table():
  with S3NetCDFAPI(input) as netcdf:
    data=_getData(netcdf,netcdf.prepareInput({"inode":[0,1],"variable":"u"}),"u")
    dimData=data['dimData']
    assert dimData2Table(data['data'],dimData)[0]=='-160.0,40.0,2000-01-01T00:00:00.000'
  

if __name__ == "__main__":
  test__getMeta()
  test_getMeta()
  test_combineValues()
  test_dimData2Table()