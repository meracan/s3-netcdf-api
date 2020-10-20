import numpy as np
from s3netcdf import NetCDF2D
from datetime import datetime
import sys
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt

def createGrid(xstart=-1,xend=1,xstep=0.1,ystart=-1,yend=1,ystep=0.1):
  xPoints = np.arange(xstart,xend+xstep,xstep)
  yPoints = np.arange(ystart,yend+ystep,ystep)
  
  xlen = len(xPoints)
  ylen = len(yPoints)

  x, y = np.meshgrid(xPoints, yPoints)
  x=x.ravel()
  y=y.ravel()
  xy=np.column_stack((x,y))
  elem = []
  for row in range(ylen-1):
    for col in range(xlen-1):
      n1 = col+row*(xlen)
      n2 = (col+1)+row*(xlen)
      n3 = col+(row+1)*(xlen)
      n4 = (col+1)+(row+1)*(xlen)
      elem.append([n1,n3,n2])
      elem.append([n2,n3,n4])
  elem =  np.array(elem)  
  
  # tri = Triangulation(x, y, elem.astype("int32"))
  # plt.triplot(tri)
  # trifinder = tri.get_trifinder()
  # plt.savefig('{}.png'.format("xxx"))
  return {"xy":xy,"elem":elem}

def main():
  grid=createGrid(-160.0,-150.0,0.1,40.0,50.0,0.1)
  x=grid['xy'][:,0]
  y=grid['xy'][:,1]
  elem=grid['elem']
  input = dict(
    name="input1",
    cacheLocation=r"../s3",
    localOnly=False,
    
    bucket="uvic-bcwave",
    cacheSize=10.0,
    ncSize=1.0,
    
    nca = dict(
      metadata=dict(title="input1",
      spatial={"x":"x","y":"y","elem":"elem","dim":"nnode"},
      temporal={"time":"time","dim":"ntime"},
      spectral={"sx":"sx","sy":"sy","dim":"nsnode","stationId":"stationid","stationName":"name"}
      ),
      dimensions = dict(
        npe=3,
        nelem=len(elem),
        nnode=len(x),
        ntime=1000,
        nstation=6,
        nchar=16,
        nsnode=10,
        nfreq=33,
        ndir=36,
      ),
      groups=dict(
        elem=dict(dimensions=["nelem","npe"],variables=dict(
          elem=dict(type="i4", units="" ,standard_name="Connectivity" ,long_name=""),
        )),
        time=dict(dimensions=["ntime"],variables=dict(
          time=dict(type="f8",units="hours since 1970-01-01 00:00:00.0" ,calendar="gregorian" ,standard_name="Datetime" ,long_name=""),
          )),
        node=dict(dimensions=["nnode"],variables=dict(
          x=dict(type="f4",units="" ,standard_name="Longitude" ,long_name=""),
          y=dict(type="f4",units="" ,standard_name="Latitude" ,long_name=""),
          )),
        snode=dict(dimensions=["nsnode"],variables=dict(
          sx=dict(type="f4",units="" ,standard_name="Longitude" ,long_name=""),
          sy=dict(type="f4",units="" ,standard_name="Latitude" ,long_name=""),
          stationid=dict(type="i4",units="" ,standard_name="Station Id" ,long_name=""),
          )),
        station=dict(dimensions=["nstation","nchar"],variables=dict(
          name=dict(type="S16",units="" ,standard_name="Station Name" ,long_name="")
          )),
        freq=dict(dimensions=["nfreq"],variables=dict(
          freq=dict(type="f4",units="Hz" ,standard_name="Frequency" ,long_name=""),
          )),
        dir=dict(dimensions=["ndir"],variables=dict(
          dir=dict(type="f4",units="radian" ,standard_name="Direction" ,long_name=""),
          )),
        s=dict(dimensions=["ntime", "nnode"] ,variables=dict(
          u=dict(type="f4",units="m/s" ,standard_name="U Velocity" ,long_name=""),
          v=dict(type="f4",units="m/s" ,standard_name="V Velocity" ,long_name=""),
          )),
        t=dict(dimensions=["nnode" ,"ntime"] ,variables=dict(
          u=dict(type="f4",units="m/s" ,standard_name="U Velocity" ,long_name=""),
          v=dict(type="f4",units="m/s" ,standard_name="V Velocity" ,long_name=""),
          )),
          
        spc=dict(dimensions=["nsnode","ntime","nfreq", "ndir"] ,variables=dict(
          spectra={"type":"f8","units":"m2/Hz/degr","standard_name": "VaDens","long_name":"variance densities in m2/Hz/degr","exception_value":-0.9900E+02}
          ))
       
      )
    )
  )
  
  netcdf2d=NetCDF2D(input)
  netcdf2d["elem","elem"] = elem
  ntime = np.prod(netcdf2d.groups["time"].shape)
  timevalue = np.datetime64(datetime(2000,1,1))+np.arange(ntime)*np.timedelta64(1, 'h')
  netcdf2d["time","time"] = timevalue.astype("datetime64[s]")
  netcdf2d["node","x"] = x
  netcdf2d["node","y"] = y
  nsnode=np.prod(netcdf2d.groups["snode"].shape)
  netcdf2d["snode","sx"] = np.arange(nsnode)*0.1-160.0
  netcdf2d["snode","sy"] = np.arange(nsnode)*0.1+40.0
  stationids=np.zeros(nsnode,dtype=np.int32)
  stationids[1]=1
  stationids[2]=2
  stationids[3:5]=3
  stationids[5:7]=4
  stationids[7:10]=5
  stationids[10:]=6
  netcdf2d["snode","stationid"] = stationids
  netcdf2d["station","name"]=np.array(['a', 'b', 'c','d','e','f'])
  
  sshape = netcdf2d.groups["s"].shape
  svalue = np.arange(np.prod(sshape)).reshape(sshape)
  
  netcdf2d["s","u"] = svalue
  netcdf2d["s","v"] = svalue
  netcdf2d["t","u"] = svalue.T
  netcdf2d["t","v"] = svalue.T
  netcdf2d["freq","freq"] = np.arange(33)/33.0
  netcdf2d["dir","dir"] = np.arange(36)/36.0
  netcdf2d["spc","spectra"] = np.arange(10*1000*33*36)/1000000.0
  
if __name__ == "__main__":
  main()