
from scipy.io import loadmat, savemat
import scipy.io as sio

def to_mat(obj,data):
  output={}
  for vname in data:
    variable=data[vname]
    output[vname]=variable['data']
    
    dimData=variable['dimData']
    if dimData is not None:
      for dName in dimData:
        _dimData=dimData[dName]
        if _dimData['data'] is None:
          subData=_dimData['subdata']
          for subdim in subData:
            output[subdim]=subData[subdim]['data']
        else:
          output[dName]=_dimData['data']
  
  
  filepath=obj['filepath']+".mat"    
  sio.savemat(filepath, output)
  return filepath
  