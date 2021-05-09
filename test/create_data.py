import sys
import os
import numpy as np
from s3netcdf import S3NetCDF
from datetime import datetime

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

def createData(localOnly=True):
  grid=createGrid(-160.0,-150.0,0.1,40.0,50.0,0.1)
  x=grid['xy'][:,0]
  y=grid['xy'][:,1]
  elem=grid['elem']
  input={
    "name":"s3netcdfapi_test",
    "cacheLocation":r"../s3",
    "localOnly":localOnly,
    "bucket":os.environ.get("AWS_BUCKETNAME",None),
    "dynamodb":os.environ.get("AWS_TABLENAME",None),
    "cacheSize":10.0,
    "ncSize":1.0,
    "overwrite":True,
    "nca":{
      "metadata":{
        "title":"s3netcdfapi_test",
        "api":{
          "dims":[
            {"id":"nnode","label":"Mesh nodes","type":"spatial"},
            {"id":"ntime","label":"Hourly timestep","type":"temporal"},
            {"id":"nsnode","label":"Spectral nodes","type":"spectral"}
            ],
          "spatials":['nnode'],
          "temporals":['ntime'],
          "spectrals":['nsnode'],
          "nnode":{"x":"x","y":"y","elem":"elem"},
          "ntime":{"time":"time"},
          "nsnode":{"x":"sx","y":"sy","stationId":"stationid","stationName":"name"},
        }
      },
      "dimensions":{"npe":3,"nelem":len(elem),"nnode":len(x),"ntime":1000,"nstation":6,"nchar":16,"nsnode":10,"nfreq":33,"ndir":36},
      "groups":{
        "elem":{"dimensions":["nelem","npe"],"variables":{
          "elem":{"type":"i4", "units":"" ,"standard_name":"Connectivity" ,"long_name":""}
          },
        },
        "time":{"dimensions":["ntime"],"variables":{
          "time":{"type":"M","standard_name":"Datetime" ,"long_name":""}
          }
        },
        "node":{"dimensions":["nnode"],"variables":{
          "x":{"type":"f4","standard_name":"Longitude" ,"long_name":""},
          "y":{"type":"f4","standard_name":"Latitude" ,"long_name":""}
          }
        },
        "snode":{"dimensions":["nsnode"],"variables":{
          "sx":{"type":"f4","standard_name":"Longitude" ,"long_name":""},
          "sy":{"type":"f4","standard_name":"Latitude" ,"long_name":""},
          "stationid":{"type":"i4","standard_name":"Station Id" ,"long_name":""}
          }
        },
        "station":{"dimensions":["nstation","nchar"],"variables":{
          "name":{"type":"S1","standard_name":"Station Name" ,"long_name":""}
          }
        },
        "freq":{"dimensions":["nfreq"],"variables":{
          "freq":{"type":"f4","units":"Hz","standard_name":"Frequency" ,"long_name":""}
          }
        },
        "dir":{"dimensions":["ndir"],"variables":{
          "dir":{"type":"f4","units":"radian","standard_name":"Direction" ,"long_name":""}
          }
        },
        "s":{"dimensions":["ntime","nnode"],"variables":{
          "u":{"type":"f4","units":"m/s","standard_name":"U Velocity" ,"long_name":""},
          "v":{"type":"f4","units":"m/s","standard_name":"V Velocity" ,"long_name":""}
          }
        },
        "t":{"dimensions":["nnode","ntime"],"variables":{
          "u":{"type":"f4","units":"m/s","standard_name":"U Velocity" ,"long_name":""},
          "v":{"type":"f4","units":"m/s","standard_name":"V Velocity" ,"long_name":""}
          }
        },
        "spc":{"dimensions":["nsnode","ntime","nfreq", "ndir"],"variables":{
          "spectra":{"type":"f8","units":"m2/Hz/degr","standard_name":"VaDens" ,"long_name":"variance densities in m2/Hz/degr","exception_value":-0.9900E+02},
          }
        },        
      }
    }
  }
  
  with S3NetCDF(input) as netcdf:
    netcdf["elem","elem"] = elem
    ntime = np.prod(netcdf.groups["time"].shape)
    timevalue = np.datetime64(datetime(2000,1,1))+np.arange(ntime)*np.timedelta64(1, 'h')
    netcdf["time","time"] = timevalue.astype("datetime64[s]")
    netcdf["node","x"] = x
    netcdf["node","y"] = y
    nsnode=np.prod(netcdf.groups["snode"].shape)
    netcdf["snode","sx"] = np.arange(nsnode)*0.1-160.0
    netcdf["snode","sy"] = np.arange(nsnode)*0.1+40.0
    stationids=np.zeros(nsnode,dtype=np.int32)
    stationids[1]=1
    stationids[2]=2
    stationids[3:5]=3
    stationids[5:7]=4
    stationids[7:10]=5
    stationids[10:]=6
    netcdf["snode","stationid"] = stationids
    netcdf["station","name"]=np.array(['a', 'b', 'c','d','e','f'])
    
    sshape = netcdf.groups["s"].shape
    svalue = np.arange(np.prod(sshape)).reshape(sshape)
    
    netcdf["s","u"] = svalue
    netcdf["s","v"] = svalue
    netcdf["t","u"] = svalue.T
    netcdf["t","v"] = svalue.T
    netcdf["freq","freq"] = np.arange(33)/33.0
    netcdf["dir","dir"] = np.arange(36)/36.0
    netcdf["spc","spectra"] = np.arange(10*1000*33*36)/1000000.0
  
if __name__ == "__main__":
  createData()