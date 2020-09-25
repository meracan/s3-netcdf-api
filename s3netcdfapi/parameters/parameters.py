import os
import uuid
import numpy as np
from .checkExport import checkExport
from .newCheck import check



def getParameters(netcdf2d,_parameters):

  pointers={"pointers":{
    "mesh":{"dimensions":["nnode"],"x":["x","lng",'longitude','lon'],"y":["y","lat","latitude"],"node":["node"]},
    "temporal":{"dimensions":["ntime"],"time":["time"]},
    "xy":{"dimensions":["nsnode"],"x":["x","sx"],"y":["y","sy"],"snode":["snode"]},
    }}
  
    
  default={
    'cache':{"default":r"../s3/tmp","type":str},
    'export':{"default":"json","type":str},
    "dataOnly":{"default":False,"type":bool},
    # 'mesh':{"default":False,"type":bool},
    'variable':{"default":None,"type":(str,list)},
    
    'inode':{"default":None,"type":(int,list,slice)},
    'isnode':{"default":None,"type":(int,list,slice)},
    'longitude':{"default":None,"type":(float,list)},
    'latitude':{"default":None,"type":(float,list)},
    'x':{"default":None,"type":(float,list)},
    'y':{"default":None,"type":(float,list)},
 
    
    'itime':{"default":None,"type":(int,list,slice)},
    'start':{"default":None,"type":str},
    'end':{"default":None,"type":str},
    'step':{"default":1,"type":int},
    'stepUnit':{"default":'h',"type":str},
    
    'inter.mesh':{"default":'closest',"type":str},
    'inter.temporal':{"default":'closest',"type":str},
    'inter.xy':{"default":'closest',"type":str},
    
    'sep':{"default":',',"type":str},
    
  }
  obj=parseParameters(default,_parameters)
  obj['filepath']=os.path.join(obj['cache'],str(uuid.uuid4()))
  obj={**pointers,**obj}
  obj=setGroups(netcdf2d,obj)
  obj=checkExport(netcdf2d,obj)
  obj=check(netcdf2d,obj)
  return obj


def setGroups(netcdf2d,obj):
  if obj['variable'] is None: obj['variable']=[]
  
  if not isinstance(obj['variable'], list):obj['variable']=[obj['variable']]
  if obj['export']=="slf" and not 'mesh' in obj['variable']:
    obj['variable'].append('mesh')
  if 'mesh' in obj['variable']:
    if not 'x' in obj['variable']:obj['variable'].append('x')
    if not 'y' in obj['variable']:obj['variable'].append('y')
    if not 'elem' in obj['variable']:obj['variable'].append('elem')
    if not 'time' in obj['variable']:obj['variable'].append('time')
    obj['variable'].remove('mesh')
    obj['inode']=None
    obj['x']=None
    obj['y']=None
    obj['latitude']=None
    obj['longitude']=None
    obj["dataOnly"]=True
    
  
    
  
  obj['groups']=groups=getGroups(netcdf2d,obj)
  obj['ngroups']=ngroups=len(groups)
  obj['isTable']= ngroups==1
  return obj
  

def getGroups(netcdf2d,obj):
  """ Get unique groups based on the variables
  """
  if len(obj['variable'])==0:return []
  _tmp=[netcdf2d.getGroupsByVariable(vname) for vname in obj['variable']]
  
  ugroups=[]
  for a in _tmp:
    con=True
    for b in ugroups:
      if con and a==b:con=False
    if con:ugroups.append(a)
  return ugroups

def formatTuple(t):
  l=list(t)
  if list in l:l.remove(list)
  if len(l)==1:return l[0]
  if float in l and int in l:return float
  if slice in l and int in l:return int
  raise Exception("Please review formatTuple {}".format(l))



def parseParameters(obj,parameters):
  newobject={}
  for o in obj:
    default=obj[o]['default']
    type=obj[o]['type']
    if o=="sep":newobject[o]=parameters.get(o,default)
    else:newobject[o]=parseParameter(parameters.get(o,default),type)
  return newobject  



def parseString(parameter,t):
  try:
    if ":" in parameter: # Test 1
      start,end=parameter.split(":")
      if start=="":start=None
      else: start=int(start)
      if end=="":end=None
      else: end=int(end)
      return slice(start,end,None)
    elif "[" in  parameter or "]" in  parameter or "," in  parameter: # Test 4
      parameter=parameter.replace("[","").replace("]","")
      parameter=parameter.split(",")
      value=list(map(lambda x:t(x),parameter))
      return value
    elif parameter.lower() in ("yes", "true"): # Test 1
      return True
    elif parameter.lower() in ("no", "false"): # Test 1
      return False
    else:
      return t(parameter) # Test 1
  except Exception as err:
    raise Exception("Format needs to be \"{string}\" or \":\" or \"{string}:{string}\" or \"[{string},{string}]\"")
  
  
def parseParameter(parameter,types=str):
  """ Parse index query based on string 
  
  Parameters
  ----------
  index:str
  
  Examples
  ---------
  
  """  
  if parameter is None:return None
  
  if type(types) is not tuple:types=(types,)
  t=formatTuple(types)
  
  if isinstance(parameter,str):parameter=parseString(parameter,t)
  if not list in list(types) and isinstance(parameter,list):
    raise Exception("Parameter {} is not allowed to be a list".format(parameter)) # Test 8
  
  
  if isinstance(parameter,list):
    if all(isinstance(i, t) for i in parameter): # Test5
      return parameter
    else:
      try: # Test6 
        value=[t(i) for i in parameter]
      except Exception as err: # Test7
        raise Exception("value={} does have the right type {}".format(parameter,t))
      return value
  else:
    if isinstance(parameter,types):return parameter
    else:
      return t(parameter)