import numpy as np
import pandas as pd
import json

from s3netcdfapi import S3NetCDFAPI
from s3netcdfapi.data.getData import _getData
from s3netcdfapi.export.table import _getMeta,getMeta,combineValues,dimData2Table
import s3netcdfapi.export as export

input={
  "name":"input1",
  "cacheLocation":"../s3",
  "apiCacheLocation":"../s3/tmp",
  "localOnly":True,
  "verbose":True,
  "maxPartitions":40
}

netcdf2d=S3NetCDFAPI(input)

def test__getMeta():
  meta={"standard_name":"Name","units":"m"}
  assert _getMeta(meta,type="header")=="Name,m"
  assert _getMeta(meta,type="units")=="m"
 
def test_getMeta():
  
  data=_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0,1],"variable":"u"}),"u")
  dimData=data['dimData']
  assert getMeta(dimData,"header")==['Datetime','Longitude','Latitude']
  assert getMeta(dimData,"type")==['float64','float32','float32']
  assert getMeta(dimData,'header',data,True)['x']=="Longitude"
  assert getMeta(dimData,'header',data,True)['y']=="Latitude"
  assert getMeta(dimData,'header',data,True)['time']=="Datetime"


def test_combineValues():
  data=_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0,1],"variable":"u"}),"u")
  dimData=data['dimData']
  assert combineValues(dimData['node']['subdata'])[0]=='-160.0,40.0'
 
 
def test_dimData2Table():
  data=_getData(netcdf2d,netcdf2d.prepareInput({"inode":[0,1],"variable":"u"}),"u")
  dimData=data['dimData']
  assert dimData2Table(data['data'],dimData)[0]=='2000-01-01T00:00:00,-160.0,40.0'
  

if __name__ == "__main__":
  test__getMeta()
  test_getMeta()
  test_combineValues()
  test_dimData2Table()