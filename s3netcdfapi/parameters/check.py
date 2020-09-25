

from .checkTemporal import checkTemporal
from .checkSpatial import checkSpatial


def check(netcdf2d,obj):
  variables=obj['variable']
  dnames=[]
  for vname in variables:
    dnames=dnames+netcdf2d.getMetaByVariable(vname)['dimensions']
  dnames=list(set(dnames))

  for dname in dnames:
    if dname in obj['pointers']['temporal']['dimensions']:
      obj=checkTemporal(netcdf2d,obj,dname)
    elif dname in obj['pointers']['mesh']['dimensions']:
      obj=checkSpatial(netcdf2d,obj,dname,'mesh')
    elif dname in obj['pointers']['xy']['dimensions']:
      obj=checkSpatial(netcdf2d,obj,dname,"xy")
    else:continue
      
  return obj