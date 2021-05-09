from .getTemporal import getTemporal
from .getSpatial import getSpatial
from .getSpectral import getSpectral


def getIndex(netcdf2d,obj):
  variables=obj['variable']
  dnames=[]
  for vname in variables:
    # dnames=dnames+netcdf2d.getMetaByVariable(vname)['dimensions']
    dnames=dnames+netcdf2d.variables[vname]['dimensions']
  dnames=list(set(dnames))

  for dname in dnames:
    if dname in netcdf2d.temporals:
      obj=getTemporal(netcdf2d,obj,dname)
    elif dname in netcdf2d.spatials:
      obj=getSpatial(netcdf2d,obj,dname,'mesh')
    elif dname in netcdf2d.spectrals:
      obj=getSpatial(netcdf2d,obj,dname,"xy")
      obj=getSpectral(netcdf2d,obj,dname)
    else:continue
      
  return obj