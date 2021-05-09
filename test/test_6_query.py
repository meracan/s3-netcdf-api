import pytest
import os
import pandas as pd
import base64
import gzip
import io
import numpy as np
import json
import time

from s3netcdfapi import S3NetCDFAPI
from s3netcdfapi.index import handler
from projects.uvicswan import uvicswan_queries


def test_query():
  input={"name":"s3netcdfapi_test","cacheLocation":"../s3","localOnly":True,"verbose":True,"maxPartitions":40,"autoRemove":False}
  with S3NetCDFAPI(input) as netcdf:
    response=netcdf.run({"id":"s3netcdfapi_test"})
    assert json.loads(response['body'])['id']=="s3netcdfapi_test"
    
    response=netcdf.run({"id":"s3netcdfapi_test","export":"csv","variable":"u,v","x":-160.0,"y":40.0,"itime":[0,1]})
    df=pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(response['body']))))
    np.testing.assert_array_equal(df['Latitude'].values,[40.0,40.0])

def test_UVICSWAN(name="SWANv5"):
  input={"name":name,"bucket":"uvic-bcwave","cacheLocation":"../s3","localOnly":False,"verbose":True,"maxPartitions":40,"autoRemove":False,}
  with S3NetCDFAPI(input) as netcdf:
    response=netcdf.run({})
    print(response)
    for query in uvicswan_queries:
      start=time.time()
      print(query)
      response=netcdf.run(query)
      assert response['statusCode']==200         
      print(time.time()-start)
      
    
class Context(object):
  def __init__(self):
    self.aws_request_id="TESTID"
context=Context()

def test_handler():
  os.environ['AWS_DEBUG']="True"
  os.environ['AWS_CACHE']=r"../s3"
  
  assert json.loads(handler({},context)['body'])['Error']=="Api needs a model id"
  assert json.loads(handler({"pathParameters":{"id":"s3netcdfapi_test"}},context)['body'])['id']=='s3netcdfapi_test'
  
  response=handler({"pathParameters":{"id":"s3netcdfapi_test"},"queryStringParameters":{"export":"csv","variable":"u,v","x":-160.0,"y":40.0,"itime":[0,1]}},context)
  df=pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(response['body']))))
  np.testing.assert_array_equal(df['Latitude'].values,[40.0,40.0])

def test_handler_UVICSWAN(name="SWANv5"):
  os.environ['AWS_DEBUG']="True"
  os.environ['AWS_CACHE']=r"../s3"
  
  assert json.loads(handler({},context)['body'])['Error']=="Api needs a model id"
  assert json.loads(handler({"pathParameters":{"id":name}},context)['body'])['id']==name
  
  response=handler({"pathParameters":{"id":name},"queryStringParameters":{"export":"csv","variable":"time"}},context)
  df=pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(response['body']))))
  print(df)
  
   
    
if __name__ == "__main__":
  test_query()
  test_handler()
  # test_UVICSWAN("SWANv5")
  # test_UVICSWAN("SWANv6")
  # test_handler_UVICSWAN("SWANv5")
  # test_handler_UVICSWAN("SWANv6")
