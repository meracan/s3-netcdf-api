import pytest
import numpy as np
from s3netcdf import NetCDF2D
from matplotlib.tri import Triangulation
import s3netcdfapi.data.interpolation as inter


input={
  "name":"input1",
  "cacheLocation":"../s3",
  "localOnly":True,
  "verbose":True
}

netcdf2d=NetCDF2D(input)

def test_timeSeriesNearest():
    # Test 1
    _datetime= np.arange('2000-01-01', '2000-02-01', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    datetime= np.arange('2000-01-10T00:15', '2000-01-11', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    data=np.arange(len(_datetime))
    np.testing.assert_array_equal(inter.timeSeries(_datetime,datetime,data),np.arange(216,240))
    
    # Test 2 - Assuming 3 time-series
    data=np.arange(len(_datetime)*3).reshape(3,len(_datetime)).T
    r2=np.append(np.arange(216,240),[np.arange(960,984),np.arange(1704,1728)]).reshape(3,24).T
    np.testing.assert_array_equal(inter.timeSeries(_datetime,datetime,data),r2)
    
    # Test 3
    _datetime= np.arange('2000-01-01', '2016-01-01', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    datetime= np.arange('2000-01-01T01:00', '2015-12-01', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    data=np.arange(len(_datetime))
    np.testing.assert_array_equal(inter.timeSeries(_datetime,datetime,data),np.arange(1,len(datetime)+1))


def test_timeSeriesLinear():
    # Test 1
    _datetime= np.arange('2000-01-01', '2000-02-01', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    datetime= np.arange('2000-01-10T00:15', '2000-01-11', np.timedelta64(1, 'h'), dtype='datetime64[s]')
    data=np.arange(len(_datetime))
    np.testing.assert_array_equal(inter.timeSeries(_datetime,datetime,data,kind="linear"),np.arange(216,240)+0.25)
    
    # Test 2 - Assuming 3 time-series
    data=np.arange(len(_datetime)*3).reshape(3,len(_datetime)).T
    r2=np.append(np.arange(216,240)+0.25,[np.arange(960,984)+0.25,np.arange(1704,1728)+0.25]).reshape(3,24).T
    np.testing.assert_array_equal(inter.timeSeries(_datetime,datetime,data,kind="linear"),r2)


def test_mesh():
    elem=netcdf2d['elem','elem'].astype("int")
    x=netcdf2d['node','x']
    y=netcdf2d['node','y']
    npoint=len(x)
    timesteps=2
    data=np.arange(npoint*timesteps).reshape((timesteps,npoint)).T
    r=inter.mesh(x,y,elem,data,x,y)
    np.testing.assert_array_almost_equal(r[:,0],np.arange(npoint),6)
    

    tri = Triangulation(x, y, elem)
    trifinder = tri.get_trifinder()
    tx=[-159.95,-159.85]
    ty=[40.0,40.0]
    ielem=trifinder.__call__(tx,ty)
   
    uielem,elemIndex=np.unique(ielem,return_inverse=True)
    idx=elem[uielem].astype("int32")
    inode,nodeIndex=np.unique(idx,return_inverse=True)
    newx=x[inode]
    newy=y[inode]
    newelem=nodeIndex.reshape(idx.shape)
 
    npoint=len(newx)
    timesteps=5000
    data=np.arange(npoint*timesteps).reshape((timesteps,npoint)).T
    
    r=inter.mesh(newx,newy,newelem,data,tx,ty)
    np.testing.assert_array_almost_equal(r[0],np.arange(0,timesteps*5,5)+0.5,4)
    np.testing.assert_array_almost_equal(r[1],np.arange(0,timesteps*5,5)+1.5,4)
  


def test_timeSeries_mesh():
    elem=netcdf2d['elem','elem'].astype("int")
    x=netcdf2d['node','x']
    y=netcdf2d['node','y']
    _datetime= netcdf2d['time','time']
    u=netcdf2d['s','u']
    
    # Test1
    datetime= np.arange('2000-01-01T00:15', '2000-01-02', np.timedelta64(1, 'h'), dtype='datetime64[s]') 
    points=np.array([[-159.95,40.0],[-159.85,40.0],[-159.75,40.0],[-159.65,40.0]])
    tx=points[:,0]
    ty=points[:,1]
    
    
    _u1=inter.mesh(x,y,elem,u.T,tx,ty)
    _u1=inter.timeSeries(_datetime,datetime,_u1.T,kind="linear")
    _u2=inter.timeSeries(_datetime,datetime,u,kind="linear")
    _u2=inter.mesh(x,y,elem,_u2.T,tx,ty).T
    np.testing.assert_almost_equal(_u1,_u2) 

  

if __name__ == "__main__":
  # test_timeSeriesNearest()
  # test_timeSeriesLinear()
  test_mesh()
  test_timeSeries_mesh()
  