import numpy as np
import pandas as pd

import csv


def table(obj,data):
  """
  """
  df = pd.DataFrame()
  for vname in data:
    _data=data[vname]
    if _data['dimData'] is None:
      _header=_data['header']
      df[_header]=_data['data'].flatten()
    else:
      if df.empty:
        f,headers=dimData2Table(_data['data'],_data['dimData'])
        df=df.from_records(csv.reader(f),columns=headers)
      df[_data['header']]=_data['data'].flatten()
  # df.to_csv('')    




def dimData2Table(data,dimData):
  """
  """
  shape=data.shape
  maxLength=0
  
  dimIndexValue=[]
  headers=[]
  for _data in dimData:
    if isinstance(_data,list):
      _d0=_data[0]['data']
      _d1=_data[1]['data']
      values=np.array(["{},{}".format(_x,_y)for _x,_y in zip(_d0,_d1)])
      headers.append(_data[0]['header'])
      headers.append(_data[1]['header'])
    else:
      values=_data['data']
      headers.append(_data['header'])
      
    values=values.astype('str')
    _max=max([len(x) for x in values])
    values=values.astype('|S{}'.format(_max))
    maxLength+=_max
    dimIndexValue.append(values)
  
  maxLength+=len(headers)
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
  return a.flatten(),headers