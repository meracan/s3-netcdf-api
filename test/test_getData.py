import pytest
from s3netcdf import NetCDF2D
import s3netcdfapi.getData as cleanObject,swapAxe,swapAxes,getData,getHeader,getDimensionValues


input={
  "name":"input1",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True
}

netcdf2d=NetCDF2D(input)


def test_cleanObject():
    None

def test_swapAxe():
    None

def test_swapAxes():
    None
    
def test_getHeader():
    None
    
def test_getDimensionValues():
    None

def test_getData():
    None

if __name__ == "__main__":
  # test_cleanObject()
  # test_swapAxe()
  # test_swapAxes()
  # test_getHeader()
  # test_getDimensionValues()
  # test_getData()
  None
