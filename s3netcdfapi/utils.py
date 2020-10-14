import gzip

def getUniqueGroups(groups,obj):
    """ Get unique groups based on the variables
    """
    if len(obj['variable'])==0:return []
    
    ugroups=[]
    for a in groups:
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
    if ":" in parameter and not "-" in parameter: # Test 1
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


def getIdx(obj,name):
  pointer=obj['pointer'][name]
  group=pointer['group']
  variable=pointer['variable']
  return (group,variable)

def compress(filePath,outPath):
  with open(filePath, 'rb') as src, gzip.open(outPath, 'wb') as dst:        
    dst.writelines(src)