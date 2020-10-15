import numpy as np
import binpy

def to_binary(netcdf2d,obj,data,min=-127,max=128):
  filepath=obj['filepath']+".bin"
  response={}
  
  x=netcdf2d.spatial['x']
  y=netcdf2d.spatial['y']
  elem=netcdf2d.spatial['elem']
  time=netcdf2d.temporal['time']
  
  for v in data:
    _data=data[v]
    d=_data['data']
    if v==x:response['x']=d
    elif v==y:response['y']=d
    elif v==elem:response['elem']=d
    elif v==time:response['time']=d
    elif len(_data['dimensions'])==1:
      response[v]=d
    else:
      dname=_data['dimensions'][0]
      if(dname=="ntime"):#Spatial
        for i,index in enumerate(_data['indices'][0]):
          id="{}_s_{}".format(v,index)
          z=np.zeros(obj['res']*obj['res']*2,dtype="uint8")
          ui=encode(d[i],min,max)
          z[:ui.size]=ui
          response[id]=z
      elif(dname=="nnode"):#Timeseries
        noderange=_data['indices'][0]
        if obj['user_xy']:noderange=list(map(lambda x:"{}_{}".format(x[0],x[1]),zip(obj['x'],obj['y'])))
        for i,index in enumerate(noderange):
          id="{}_t_{}".format(v,index)
          response[id]=d[i]        
      
      elif(dname=="nsnode"):#Spectra
        noderange=_data['indices'][0]
        timerange=_data['indices'][1]
        
        if obj['user_xy']:noderange=list(map(lambda x:"{}_{}".format(x[0],x[1]),zip(obj['x'],obj['y'])))
        if obj['user_time']:timerange=list(map(lambda x:"{}".format(x),obj['time']))
  
        for i,indexi in enumerate(noderange):
          for j,indexj in enumerate(timerange):
            id="{}_{}_{}".format(v,indexi,indexj)
            response[id]=d[i,j]
      else:
        for i,index in enumerate(_data['indices'][0]):
          id="{}_{}_{}".format(v,dname,index)
          response[id]=d[i]
  
  binpy.write(response,filepath)
  return filepath

def encode(array,min,max):
  value = (array - min) / (max - min) * 255 * 255
  
  return np.column_stack((np.round(np.remainder(value, 255.0)),np.floor(value / 255.0))).astype("uint8").flatten()

def decode(pair,min,max):
    y = pair[:,0]
    x = pair[:,1] * 255.0;
    return (y + x) / 255.0 / 255.0 * (max - min) + min;
  