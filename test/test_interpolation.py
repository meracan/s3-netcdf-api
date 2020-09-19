import numpy as np
import s3netcdfapi.interpolation as inter
from s3netcdfapi.parameters import getParameters
from s3netcdf import NetCDF2D
import pytest

input={
  "name":"input1",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True
}

netcdf2d=NetCDF2D(input)

def test_timeSeries():
    # Test 1
    _datetime= np.arange('2000-01-01', '2000-02-01', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    datetime= np.arange('2000-01-10T00:15', '2000-01-11', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    data=np.arange(len(_datetime))
    np.testing.assert_array_equal(inter.timeSeries(_datetime,datetime,data),np.arange(216,240)+0.25)
    
    # Test 2 - Assuming 3 time-series
    data=np.arange(len(_datetime)*3).reshape(3,len(_datetime)).T
    r2=np.append(np.arange(216,240)+0.25,[np.arange(960,984)+0.25,np.arange(1704,1728)+0.25]).reshape(3,24).T
    np.testing.assert_array_equal(inter.timeSeries(_datetime,datetime,data),r2)

def test_barycentric():
    elem=netcdf2d['elem','elem'].astype("int")
    x=netcdf2d['node','x']
    y=netcdf2d['node','y']
    npoint=len(x)
    
    r1=inter.barycentric(elem,x,y,np.array([[-159.95,40.0],[-159.85,40.0],[-159.75,40.0],[-159.65,40.0]]),np.arange(len(x)))
    
    array=np.tile(np.array([-159.95,40.0]), (1000, 1))
    _referenceData=np.arange(npoint*10000).reshape((10000,npoint)).T
    r2=inter.barycentric(elem,x,y,array,_referenceData)
    print(r2)

def test_timeSeries_barycentric():
    print(netcdf2d.query({"variable":'x'},True))
    elem=netcdf2d['elem','elem'].astype("int")
    x=netcdf2d['node','x']
    y=netcdf2d['node','y']
    _datetime= np.arange('2000-01-01', '2000-02-01', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    u=netcdf2d['s','u']
    
    # Test1
    datetime= np.arange('2000-01-01T00:15', '2000-01-02', np.timedelta64(1, 'h'), dtype='datetime64[s]') 
    points=np.array([[-159.95,40.0],[-159.85,40.0],[-159.75,40.0],[-159.65,40.0]])
    
    _u1=inter.barycentric(elem,x,y,points,u.T)
    _u1=inter.timeSeries(_datetime,datetime,_u1.T)
    _u2=inter.timeSeries(_datetime,datetime,u)
    _u2=inter.barycentric(elem,x,y,points,_u2.T).T
    np.testing.assert_almost_equal(_u1,_u2) 

def test_getInode():
  obj={
      'meshx':None,'meshy':None,'elem':None,
      'x':[-159.95,-159.85,-159.75,-159.65],
      'y':[40.0,40.0,40.0,40.0],
      'pointer':{      
        "meshx":{"group":"node",'variable':'x'},
        "meshy":{"group":"node",'variable':'y'},
        "elem":{"group":"elem",'variable':'elem'}}}
  inter.getInode(netcdf2d,obj,'u')

def test_IDW():
    # Test 1
    parameters1={'variable':'u','longitude':[-159.9,-159.9],'latitude':[40.0,40.0],'itime':0}
    obj1=getParameters(netcdf2d,parameters1)
    print(inter.IDW(netcdf2d,obj1,'u'))
    
# def test_linear():
#     # Test 1
#     parameters1={'variable':'u','longitude':[-159.95,-159.85,-159.75],'latitude':[40.0,40.0,40.0],'itime':0}
#     obj1=getParameters(netcdf2d,parameters1)
#     print(inter.linear(netcdf2d,obj1,'u'))
    

def test_closest():
    # Test 1
    i=list(np.arange(1,dtype="int"))
    parameters1={'variable':'u','longitude':[-159.95,-159.85,-159.75],'latitude':[40.0,40.0,40.0],'itime':i}
    
    obj1=getParameters(netcdf2d,parameters1)
    print(inter.closest(netcdf2d,obj1,'u').shape)
    

def test_tlinear():
    inter.tlinear()
    

def test_tclosest():
    inter.tclosest()
    

if __name__ == "__main__":
    # test_timeSeries()
  # test_barycentric()
  test_timeSeries_barycentric()
  # test_getInode()
  
#   test_IDW()
  
#   test_closest()
#   test_tlinear()
#   test_tclosest()