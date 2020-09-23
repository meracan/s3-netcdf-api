import numpy as np
import pandas as pd
import json
import csv

def to_table(obj,data,return_xy_header=False):
  """
  """
  df = pd.DataFrame()
  xyHeaders={}
  for vname in data:
    variable=data[vname]
    _header=_getMeta(variable['meta'],'header')
    _data=variable['data']
    _dimData=variable['dimData']
    if _dimData is None:
      df[_header]=_data.flatten()
    else:
      if df.empty:
        f=dimData2Table(_data,_dimData)
        headers=getMeta(_dimData,"header")
        types=getMeta(_dimData,"type")
        df=df.from_records(csv.reader(f),columns=headers)
        for i,id in enumerate(headers):
          if id!="Datetime":df[id]=df[id].astype(types[i])
      df[_header]=_data.flatten()
    xyHeaders={**xyHeaders,**getMeta(_dimData,'header',variable,True)}
    
  
  if not return_xy_header:return df
  return df,xyHeaders    


def dimData2Table(data,dimData):
  """
  """
  shape=data.shape
  maxLength=0
  dimIndexValue=[]

  for _data in dimData:
    if isinstance(_data,list):
      _d0=_data[0]['data']
      _d1=_data[1]['data']
      values=np.array(["{},{}".format(_x,_y)for _x,_y in zip(_d0,_d1)])
    else:
      values=_data['data']

      
    values=values.astype('str')
    _max=max([len(x) for x in values])
    values=values.astype('|S{}'.format(_max))
    maxLength+=_max
    dimIndexValue.append(values)
  
  maxLength+=len(getMeta(dimData,"header"))
  a=np.chararray((np.prod(shape)), itemsize=maxLength).reshape(shape)
  a[:]=""
  
  if len(shape)!=len(dimIndexValue):raise Exception("Shape of dimension index values does not match the data")
  for i,(ishape,indexValue) in enumerate(zip(shape,dimIndexValue)):
    if ishape!=len(indexValue):raise Exception("Error here {},{}".format(ishape,indexValue))    
    
    t=[slice(None)for j in range(i)]
    for k in range(ishape):
        _t=tuple(list(t)+[k])
        if isinstance(a[_t],str):a[_t]=a[_t].encode()+indexValue[k]+b","
        else:a[_t]=a[_t]+indexValue[k]+b","
        

  a = np.char.strip(a.astype('<U{}'.format(maxLength)), ',')
  return a.flatten()


def getMeta(dimData=None,type="header",data=None,obj=None):
  """
  """
  array=[]
  names=[]
  if data is not None:
    names.append(data['name'])
    array.append(_getMeta(data['meta'],type))
  
  if dimData is not None:  
    for _data in dimData:
      if not isinstance(_data,list):_data=[_data]
      for _d in _data:
        names.append(_d['name'])
        array.append(_getMeta(_d['meta'],type))
      
  if obj is None:return array
  obj={}
  for name,value in zip(names,array):
    obj[name]=value
  return obj
  

def _getMeta(meta,type="header"):
  """
  """
  
  value=""
  if type=="header":
    value=meta['standard_name']
    if value!="Datetime" and meta['units']!="":value="{},{}".format(value,meta['units'])
  else:
    value=meta[type]
  
  return value