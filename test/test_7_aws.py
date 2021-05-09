import pytest
import os

import pandas as pd
import base64
import gzip
import io
import numpy as np
import json

from s3netcdfapi import S3NetCDFAPI
from s3netcdfapi.index import handler
from create_data import createData

os.environ['AWS_DEBUG']="False"
os.environ['AWS_CACHE']=r"../s3"
os.environ['AWS_TABLENAME']=r"cccris-data"

def create_data():
  os.environ['AWS_BUCKETNAME']="cccris"
  createData(False)

class Context(object):
  def __init__(self):
    self.aws_request_id="TESTID"
context=Context()  

def test_handler():
  assert json.loads(handler({},context)['body'])['Error']=="Api needs a model id"
  assert json.loads(handler({"pathParameters":{"id":"s3netcdfapi_test"}},context)['body'])['id']=='s3netcdfapi_test'
  response=handler({"pathParameters":{"id":"s3netcdfapi_test"},"queryStringParameters":{"export":"csv","variable":"u,v","x":-160.0,"y":40.0,"itime":[0,1]}},context)
  df=pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(response['body']))))
  np.testing.assert_array_equal(df['Latitude'].values,[40.0,40.0])
  

if __name__ == "__main__":
  # create_data()
  test_handler()
  