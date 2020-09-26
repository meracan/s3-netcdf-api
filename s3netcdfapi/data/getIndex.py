from .getTemporal import getTemporal
from .getSpatial import getSpatial


def getIndex(netcdf2d,obj):
  variables=obj['variable']
  dnames=[]
  for vname in variables:
    dnames=dnames+netcdf2d.getMetaByVariable(vname)['dimensions']
  dnames=list(set(dnames))

  for dname in dnames:
    if dname in obj['pointers']['temporal']['dimensions']:
      obj=getTemporal(netcdf2d,obj,dname)
    elif dname in obj['pointers']['mesh']['dimensions']:
      obj=getSpatial(netcdf2d,obj,dname,'mesh')
    elif dname in obj['pointers']['xy']['dimensions']:
      obj=getSpatial(netcdf2d,obj,dname,"xy")
    else:continue
      
  return obj