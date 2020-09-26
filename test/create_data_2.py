import numpy as np
from s3netcdf import NetCDF2D
from datetime import datetime
import sys
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt

variables={
  "u10": {
    "type":"f4",
    "units": "m/s", 
    "standard name": "u Wind velocity", 
    "long name": "u Wind velocity, m/s",
    "matfile name": "Windv_x"
  },
  "v10": {
    "type":"f4",
    "units": "m/s", 
    "standard name": "v Wind velocity", 
    "long name": "v Wind velocity, m/s",
    "matfile name": "Windv_y"
  },
  "hs": {
    "type":"f4",
    "units": "m", 
    "standard name": "significant wave height", 
    "long name": "significant wave height (in s)",
    "matfile name": "Hsig"
  },
  "tps": {
    "type":"f4",
    "units": "s", 
    "standard name": "peak period", 
    "long name": "smoothed peak period (in s)",
    "matfile name": "TPsmoo"
  },
  "tmm10": {
    "type":"f4",
    "units": "s", 
    "standard name": "mean absolute wave period", 
    "long name": "mean absolute wave period (in s)",
    "matfile name": "Tm_10"
  },
  "tm01": {
    "type":"f4",
    "units": "s", 
    "standard name": "mean absolute wave period", 
    "long name": "mean absolute wave period (in s)",
    "matfile name": "Tm01"
  },
  "tm02": {
    "type":"f4",
    "units": "s", 
    "standard name": "mean absolute zero-crossing period", 
    "long name": "mean absolute zero-crossing period (in s)",
    "matfile name": "Tm02"
  },
  "pdir": {
    "type":"f4",
    "units": "degrees", 
    "standard name": "peak wave direction", 
    "long name": "peak wave direction in degrees",
    "matfile name": "PkDir"
  },
  "dir": {
    "type":"f4",
    "units": "", 
    "standard name": "mean wave direction", 
    "long name": "mean wave direction (Cartesian or Nautical convention)",
    "matfile name": "Dir"
  },
  "dspr": {
    "type":"f4",
    "units": "degrees", 
    "standard name": "directional wave spread", 
    "long name": "directional spreading of the waves (in degrees)",
    "matfile name": "Dspr"
  },
  "qp": {
    "type":"f4",
    "units": "", 
    "standard name": "peakedness of wave spectrum",
    "long name": "peakedness of the wave spectrum (dimensionless)",
    "matfile name": "Qp"
  },
  "transpx": {
    "type":"f4",
    "units": "m3/s", 
    "standard name": "x transport of energy",
    "long name": "x transport of energy (in W/m or m3/s)",
    "matfile name": "Transp_x"
  },
  "transpy": {
    "type":"f4",
    "units": "m3/s",
    "standard name": "y transport of energy",
    "long name": "y transport of energy (in W/m or m3/s)",
    "matfile name": "Transp_y"
  }
}
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

  return {"xy":xy,"elem":elem}

def main():
  grid=createGrid(0.0,10.0,0.1,0.0,10.0,0.1)
  x=grid['xy'][:,0]
  y=grid['xy'][:,1]
  elem=grid['elem']
  input = dict(
    name="input2",
    cacheLocation=r"../s3",
    localOnly=True,
    
    bucket="uvic-bcwave",
    cacheSize=10.0,
    ncSize=1.0,

    nca = dict(
      metadata=dict(title="input2"),
      dimensions = dict(
        npe=3,
        nelem=len(elem),
        nnode=len(x),
        ntime=1000,
        nfeature=6,
        nsnode=10,
        nfreq=33,
        ndir=36,
        nchar=32
      ),
      groups=dict(
        elem=dict(dimensions=["nelem","npe"],variables=dict(
          elem=dict(type="i4", units="" ,standard_name="Connectivity" ,long_name=""),
        )),
        time=dict(dimensions=["ntime"],variables=dict(
          time=dict(type="f8",units="hours since 1970-01-01 00:00:00.0" ,calendar="gregorian" ,standard_name="Datetime" ,long_name=""),
          )),
        node=dict(dimensions=["nnode"],variables=dict(
          lon=dict(type="f4",units="" ,standard_name="Longitude" ,long_name=""),
          lat=dict(type="f4",units="" ,standard_name="Latitude" ,long_name=""),
          bed=dict(type="f4",units="" ,standard_name="Latitude" ,long_name=""),
          )),
        feature=dict(dimensions=["nfeature","nchar"],variables=dict(
          name=dict(type="S2",units="" ,standard_name="Feature Name" ,long_name=""),
          )),
        snode=dict(dimensions=["nsnode"],variables=dict(
          slon=dict(type="f4",units="" ,standard_name="Longitude" ,long_name=""),
          slat=dict(type="f4",units="" ,standard_name="Latitude" ,long_name=""),
          feature=dict(type="i4",units="" ,standard_name="Feature Id" ,long_name=""),
          )),          
        freq=dict(dimensions=["nfreq"],variables=dict(
          freq=dict(type="f4",units="Hz" ,standard_name="Frequency" ,long_name=""),
          )),
        dir=dict(dimensions=["ndir"],variables=dict(
          dir=dict(type="f4",units="radian" ,standard_name="Direction" ,long_name=""),
          )),
        s=dict(dimensions=["ntime", "nnode"] ,variables=variables),
        t=dict(dimensions=["nnode" ,"ntime"] ,variables=variables),
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
  nsnode=np.prod(netcdf2d.groups["snode"].shape)
  feature=np.zeros(nsnode,dtype=np.int32)
  feature[1]=1
  feature[2]=2
  feature[3:6]=3
  feature[6]=4
  feature[7]=5
  feature[8]=5
  feature[9]=6
  sshape = netcdf2d.groups["s"].shape
  svalue = np.arange(np.prod(sshape)).reshape(sshape)
  
  netcdf2d["time","time"] = timevalue.astype("datetime64[s]")
  netcdf2d["node","lon"] = x
  netcdf2d["node","lat"] = y
  netcdf2d["snode","slon"] = np.arange(nsnode)*0.1
  netcdf2d["snode","slat"] = np.arange(nsnode)*0.1
  netcdf2d["snode","feature"] = feature
  netcdf2d["feature","name"] = ["beverly","brooks","c_dixon","c_eliz","campbell","e_dell"]
  netcdf2d["freq","freq"] = np.arange(33)/33.0
  netcdf2d["dir","dir"] = np.arange(36)/36.0
  netcdf2d["spc","spectra"] = np.arange(10*1000*33*36)/1000000.0
  
  for var in ['u10','v10','hs','tps','tmm10','tm01','tm02','pdir','dspr','qp','transpx','transpy']:
    netcdf2d["s",var] = svalue
    netcdf2d["t",var] = svalue.T
  
  
if __name__ == "__main__":
  main()