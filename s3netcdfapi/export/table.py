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
    _header=_getMeta(variable,'header')
    _data=variable['data']
    _dimData=variable['dimData']
    if _dimData is None:
      df[_header]=_data.flatten()
    else:
      if df.empty:
        f=dimData2Table(_data,_dimData)
        headers=getMeta(_dimData,"header")
        types=getMeta(_dimData,"type")
        calendars=getMeta(_dimData,"calendar")
        df=df.from_records(csv.reader(f),columns=headers)
        
        for header,type,calendar in zip(headers,types,calendars):
          if not calendar:df[header]=df[header].astype(type)
          
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

  for dim in dimData:
    _dimData=dimData[dim]
    _data=_dimData['data']
    if _data is None:
      values=combineValues(_dimData['subdata'])
    else:
      values=_data
      
    values=values.astype('S')
    maxLength+=values.dtype.itemsize
    dimIndexValue.append(values)
  
  maxLength+=len(getMeta(dimData,"header"))
  a=np.chararray((np.prod(shape)), itemsize=maxLength).reshape(shape)
  a[:]=""
  if len(shape)!=len(dimIndexValue):raise Exception("Shape of dimension index values does not match the data")
  for i,(ishape,indexValue) in enumerate(zip(shape,dimIndexValue)):
    if ishape!=len(indexValue):raise Exception("Error here {},{},{}".format(i,ishape,len(indexValue)))    
    
    t=[slice(None)for j in range(i)]
    for k in range(ishape):
        _t=tuple(list(t)+[k])
        if isinstance(a[_t],str):a[_t]=a[_t].encode()+indexValue[k]+b","
        else:a[_t]=a[_t]+indexValue[k]+b","

  a = np.char.strip(a.astype('<U{}'.format(maxLength)), ',')
  return a.flatten()


def combineValues(subdata):
  """
  """
 
  maxLength=len(subdata.keys())*15+len(subdata.keys())
  nvalue=len(subdata[list(subdata.keys())[0]]['data'])
  
  c=np.chararray(nvalue, itemsize=maxLength)
  c[:]=""
  for subdim in subdata:
    _data=subdata[subdim]['data']
    c=c+_data.astype("S")+b","

  c=np.char.strip(c.astype('U'), ',')
  return c


def getMeta(dimData=None,type="header",data=None,obj=None):
  """
  """
  values=[]
  names=[]
  if data is not None:
    names.append(data['name'])
    values.append(_getMeta(data,type))
  
  if dimData is not None:  
    for dname in dimData:
      _data=dimData[dname]
      if _data['data'] is None:
        for _dname in _data['subdata']:
          subdata= _data['subdata'][_dname]
          names.append(_dname)
          values.append(_getMeta(subdata,type))
      else:
        names.append(dname)
        values.append(_getMeta(_data,type))
      
      
  if obj is None:return values
  obj={}
  for name,value in zip(names,values):
    obj[name]=value
  return obj
  

def _getMeta(data,type="header"):
  """
  """
  name=data['name']
  meta=data['meta']
  
  value=""
  if type=="header":
    if "standard_name" in meta:value=meta['standard_name']
    else:value=name
    if not 'calendar' in meta and "units" in meta and meta['units']!="" :value="{},{}".format(value,meta['units'])
  else:
    if type in meta:value=meta[type]
    else:value=None
  
  return value