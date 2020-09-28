import pytest
import numpy as np
from s3netcdf import NetCDF2D
import s3netcdfapi.data.interpolation as inter


input={
  "name":"input1",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True
}

netcdf2d=NetCDF2D(input)

def test_timeSeriesClosest():
    # Test 1
    _datetime= np.arange('2000-01-01', '2000-02-01', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    datetime= np.arange('2000-01-10T00:15', '2000-01-11', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    data=np.arange(len(_datetime))
    np.testing.assert_array_equal(inter.timeSeriesClosest(_datetime,datetime,data),np.arange(216,240))
    
    # Test 2 - Assuming 3 time-series
    data=np.arange(len(_datetime)*3).reshape(3,len(_datetime)).T
    r2=np.append(np.arange(216,240),[np.arange(960,984),np.arange(1704,1728)]).reshape(3,24).T
    np.testing.assert_array_equal(inter.timeSeriesClosest(_datetime,datetime,data),r2)


def test_timeSeriesLinear():
    # Test 1
    _datetime= np.arange('2000-01-01', '2000-02-01', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    datetime= np.arange('2000-01-10T00:15', '2000-01-11', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    data=np.arange(len(_datetime))
    np.testing.assert_array_equal(inter.timeSeriesLinear(_datetime,datetime,data),np.arange(216,240)+0.25)
    
    # Test 2 - Assuming 3 time-series
    data=np.arange(len(_datetime)*3).reshape(3,len(_datetime)).T
    r2=np.append(np.arange(216,240)+0.25,[np.arange(960,984)+0.25,np.arange(1704,1728)+0.25]).reshape(3,24).T
    np.testing.assert_array_equal(inter.timeSeriesLinear(_datetime,datetime,data),r2)

def test_barycentric():
    elem=netcdf2d['elem','elem'].astype("int")
    x=netcdf2d['node','x']
    y=netcdf2d['node','y']
    npoint=len(x)
    
    # Test 1 - 1 Frame
    r1=inter.barycentric(elem,x,y,np.array([[-159.95,40.0],[-159.85,40.0],[-159.75,40.0],[-159.65,40.0]]),np.arange(len(x)))
    np.testing.assert_array_almost_equal(r1,[0.5,1.5,2.5,3.5],4)
    
    # Test 2 - 2 Frames
    array=np.array([[-159.95,40.0]])
    _referenceData=np.arange(npoint*2).reshape((2,npoint)).T
    r2=inter.barycentric(elem,x,y,array,_referenceData)
    np.testing.assert_array_almost_equal(r2,[[0.5,10302.5]],4)
    
    # Performance test
    # array=np.tile(np.array([-159.95,40.0]), (1000, 1))
    # _referenceData=np.arange(npoint*10000).reshape((10000,npoint)).T
    # r2=inter.barycentric(elem,x,y,array,_referenceData)
    # print(r2)

def test_timeSeries_barycentric():
    elem=netcdf2d['elem','elem'].astype("int")
    x=netcdf2d['node','x']
    y=netcdf2d['node','y']
    _datetime= np.arange('2000-01-01', '2000-02-01', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    u=netcdf2d['s','u']
    
    # Test1
    datetime= np.arange('2000-01-01T00:15', '2000-01-02', np.timedelta64(1, 'h'), dtype='datetime64[s]') 
    points=np.array([[-159.95,40.0],[-159.85,40.0],[-159.75,40.0],[-159.65,40.0]])
    
    _u1=inter.barycentric(elem,x,y,points,u.T)
    _u1=inter.timeSeriesLinear(_datetime,datetime,_u1.T)
    _u2=inter.timeSeriesLinear(_datetime,datetime,u)
    _u2=inter.barycentric(elem,x,y,points,_u2.T).T
    np.testing.assert_almost_equal(_u1,_u2) 

  

if __name__ == "__main__":
  test_timeSeriesClosest()
  test_timeSeriesLinear()
  test_barycentric()
  test_timeSeries_barycentric()