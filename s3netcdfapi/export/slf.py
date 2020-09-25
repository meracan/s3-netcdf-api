import numpy as np
from slfpy import SLF

def to_slf(obj,data):
  x=data['x']['data']
  y=data['y']['data']
  elem=data['elem']['data']
  dt=data['time']['data']
  slf=SLF()
  slf.addTITLE("")
  slf.addPOIN(np.column_stack((x,y)))
  slf.addIKLE(elem)
  minDate=np.min(dt)
  # TODO minDate
  seconds=(dt-minDate).astype("timedelta64[s]")
  slf.tags['times']=seconds
  
  del data['x']
  del data['y']
  del data['elem']
  del data['time']
  
  array=None
  for vname in data:
    slf.addVAR({"name":data[vname]['meta']['standard_name'],"unit":data[vname]['meta']['units']})
    if array is None:array=data[vname]['data']
    else:array=np.stack((array,data[vname]['data']))
  slf.values=np.einsum('ijk->jik', array)
  filepath=obj['filepath']+".slf"
  slf.write(filepath)
  return filepath