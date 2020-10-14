import binpy

def to_binary(netcdf2d,obj,_data):
  filepath=obj['filepath']+".bin"
  data={}
  name=netcdf2d.name
  x=netcdf2d.spatial['x']
  y=netcdf2d.spatial['y']
  elem=netcdf2d.spatial['elem']
  time=netcdf2d.temporal['time']
  
  # sdim=netcdf2d.spatial['dim']
  # tdim=netcdf2d.temporal['dim']
  # udim=netcdf2d.spectral['dim']
  
  for v in _data:
    d=_data[v]['data']
    if v==x:data['x']=d
    elif v==y:data['y']=d
    elif v==elem:data['elem']=d
    elif v==time:data['time']=d
    elif len(_data[v]['dimensions'])==1:
      data[v]=d
    else:
      dname=_data[v]['dimensions'][0]
      for i,index in enumerate(_data[v]['indices'][0]):
        id="{}_{}_{}".format(v,dname,index)
        data[id]=d[i]
    
  
  binpy.write(data,filepath)
  return filepath