import numpy as np
import base64

from s3netcdf import NetCDF2D
from s3netcdfapi.save import saveJSON,saveBinary
import binarypy

def test_JSON():
  swan=NetCDF2D({"name":"test1","bucket":"uvic-bcwave","localOnly":True,"cacheLocation":"../../s3"})
  data={
    "bed":swan["nodes","bed"]  
  }
  body=saveJSON(data)
  assert body =="{\"bed\": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]}"
  
def test_Binary():
  swan=NetCDF2D({"name":"test1","bucket":"uvic-bcwave","localOnly":True,"cacheLocation":"../../s3"})
  data={
    "bed":swan["nodes","bed"]  
  }
  body=saveBinary(data)
  body=base64.b64decode(body)
  checkData=binarypy.read(body)
  np.testing.assert_array_equal(checkData['bed'], np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
  
  
 
  
  

if __name__ == "__main__":
  test_JSON()
  test_Binary()