from .getTemporal import getTemporal
from .getSpatial import getSpatial
from .getSpectral import getSpectral


def getIndex(netcdf2d,obj):
  variables=obj['variable']
  dnames=[]
  for vname in variables:
    dnames=dnames+netcdf2d.getMetaByVariable(vname)['dimensions']
  dnames=list(set(dnames))

  for dname in dnames:
    if dname==netcdf2d.temporal['dim']:
      obj=getTemporal(netcdf2d,obj,dname)
    elif dname==netcdf2d.spatial['dim']:
      obj=getSpatial(netcdf2d,obj,dname,'mesh')
    elif dname==netcdf2d.spectral['dim']:
      obj=getSpatial(netcdf2d,obj,dname,"xy")
      obj=getSpectral(netcdf2d,obj,dname)
    else:continue
      
  return obj