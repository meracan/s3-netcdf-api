

template={
  # "spatial":["nnode","nsnode"], x,y,lat,lng,extent,etc
  # "temporal":["ntime"], start,end,step,
  
  "nnode":{"vnames":["x","y","bed"],"iselect":"inode"},
  "ntime":{"vnames":["time"],"iselect":"itime",}
  
}

def check(netcdf2d,obj,variables):
  groups=[]
  for variable in variables:
      groups=groups+netcdf2d.getMetaByVariable(variable)['dimensions']
  
  # unique
  for shape in groups:
    for dim in shape:
      obj=getParameters(obj,dim)

def getParameters(obj,dim):
  vnames=getVariablesByDimension(dim)
  template[dim]={"vnames":vnames,"iselect":"i{}".format(dim[1:])} # x,y,bed,etc
  
  if dim in template['temporal']:return getTemporal(obj,dim)
  elif dim in template['spatial']:return getSpatial(obj,dim)
  
        
def getTemporal(obj,dim):
  
  
def getSpatial(obj,dim):
  None