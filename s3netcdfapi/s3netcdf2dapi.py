import os
import boto3
import uuid
import json
import numpy as np
from s3netcdf import NetCDF2D
from .data import getIndex,getData
from .utils import parseParameters,getUniqueGroups
from .response import response,responseSignedURL
from .export import export

class S3NetCDFAPI(NetCDF2D):
  def __init__(self, obj):
    super().__init__(obj)
    self.pointers={
    "mesh":{"dimensions":["nnode"],"x":["x","lng",'longitude','lon'],"y":["y","lat","latitude"],"node":["node"]},
    "temporal":{"dimensions":["ntime"],"time":["time"]},
    "xy":{"dimensions":["nsnode"],"x":["x","sx","slon"],"y":["y","sy","slat"],"snode":["snode"]},
    }
    
    
  @staticmethod  
  def checkNetCDFExist(credentials,id,prefix,bucket):
    s3 = boto3.client('s3',**credentials)
    _prefix= "" if prefix is None else prefix+"/"
    key="{0}{1}/{1}.nca".format(_prefix,id)
    # TODO: Try and catch
    # Catch, get list of models
    s3.head_object(Bucket=bucket, Key=key)# Check if object exist, if not returns Exception
  
  
  @staticmethod
  def init(parameters,credentials):
    id = parameters.pop("id",os.environ.get("AWS_DEFAULTMODEL",None))
    if id is None:raise Exception("Api needs a model id")
    isDebug=os.environ.get('AWS_DEBUG',"True")
    if isDebug=="True":
      bucket=os.environ.get('AWS_BUCKETNAME',"uvic-bcwave")
      bucket=parameters.get('bucket',bucket)
      prefix = os.environ.get("AWS_PREFIX",None)
      localOnly = parameters.get("localOnly",True)
      print(prefix)
      netcdf2d=S3NetCDFAPI({"name":id,"s3prefix":prefix,"bucket":bucket,"verbose":True,"localOnly":localOnly,"cacheLocation":r"../s3","apiCacheLocation":r"../s3/tmp","credentials":credentials})
    else:
      bucket=os.environ.get('AWS_BUCKETNAME',None)
      prefix = os.environ.get("AWS_PREFIX",None)
      cache=os.environ.get('AWS_CACHE','/tmp')
      S3NetCDFAPI.checkNetCDFExist(credentials,id,prefix,bucket)
      netcdf2d=S3NetCDFAPI({"name":id,"prefix":prefix,"bucket":bucket,"localOnly":False,"cacheLocation":cache,"apiCacheLocation":cache})
    return netcdf2d
  
    
  def getVariableByDimension(self,dname,pointer,pointername):
    """
    """
    vnames=self.getVariablesByDimension(dname)
    vname=next(x for x in vnames if x in pointer[pointername])
    return vname
  
  
  def getDefaultParameters(self):
    """
    """

    dimensions=self._meta['dimensions']
    dimensions.pop('npe',None)
    dimensions.pop('nchar',None)
    parameters={}
   
    for dname in dimensions:
      idname="i"+dname[1:]
      parameters[idname]={"default":None,"type":(int,list,slice),"nvalue":dimensions[dname],"comment":"Index value of {}".format(dname)}
    
    x=self.getVariableByDimension('nnode',self.pointers['mesh'],'x')
    y=self.getVariableByDimension('nnode',self.pointers['mesh'],'y')
  
    parameters["variable"]={"default":None,"type":(str,list),"comment":"name of variables"}
    parameters['x']={"default":None,"type":(float,list),"comment":"Interpolate data by specifying the {}".format(x)}
    parameters['y']={"default":None,"type":(float,list),"comment":"Interpolate data by specifying the {}".format(y)}
    parameters['longitude']={"default":None,"type":(float,list),"comment":"Interpolate data by specifying the {}".format(x)}
    parameters['latitude']={"default":None,"type":(float,list),"comment":"Interpolate data by specifying the {}".format(y)}    
    parameters["start"]={"default":None,"type":str,"comment":"Startdate (yyyy-mm-ddThh:mm:ss)"}
    parameters["end"]={"default":None,"type":str,"comment":"Endate (yyyy-mm-ddThh:mm:ss)"}
    parameters["step"]= {"default":1,"type":int,"comment":"Timestep(integer)"}
    parameters["stepUnit"]={"default":"h","type":str,"comment":"Timestep unit(s,h,d,w)"}
    parameters["inter.mesh"]={"default":"nearest","type":str,"values":["nearest","linear"],"comment":"Timestep unit(s,h,d,w)"}
    parameters["inter.temporal"]={"default":"nearest","type":str,"values":["nearest","linear"],"comment":"Type of spatial interpolation"}
    parameters["inter.xy"]={"default":"nearest","type":str,"values":["nearest"],"comment":"Type of spatial interpolation"}
    parameters["export"]={"default":"json","type":str}
    parameters["sep"]={"default":",","type":str}
  
    return parameters


  def getDefaultParametersExtra(self):
    parameters=self.getDefaultParameters()
    
    variables=self.getVariables()
    vnames=list(variables.keys())    
    parameters['variable']=vnames
    
    xname=self.getVariableByDimension('nnode',self.pointers['mesh'],'x')
    yname=self.getVariableByDimension('nnode',self.pointers['mesh'],'y')
    x=self.query({'variable':xname})
    y=self.query({'variable':yname})
    parameters[xname]['extent']=[float(np.min(x)),float(np.max(x))]
    parameters[yname]['extent']=[float(np.min(y)),float(np.max(y))]
    
    timename=self.getVariableByDimension('ntime',self.pointers['temporal'],'time')
    time=self.query({'variable':timename})
    parameters['start']['minDate']=np.datetime_as_string(np.min(time), unit='s')
    parameters['start']['maxDate']=np.datetime_as_string(np.max(time), unit='s')
    parameters['end']['minDate']=np.datetime_as_string(np.min(time), unit='s')
    parameters['end']['maxDate']=np.datetime_as_string(np.max(time), unit='s') 
    
    for v in parameters:
      if 'type' in parameters[v]:
        if isinstance(parameters[v]['type'],tuple):
           parameters[v]['type']=",".join([a.__name__ for a in list(parameters[v]['type'])])
        else:
          parameters[v]['type']=parameters[v]['type'].__name__
    
    return parameters
  
  def getParameters(self,parameters):
    return {"pointers":self.pointers,**parseParameters(self.getDefaultParameters(),parameters)}
  
  def prepareInput(self,parameters):
    """
    """
    obj=self.getParameters(parameters)
    obj=self.checkParameters(obj)
    obj=self.checkExport(obj)
    obj=getIndex(self,obj)
    return obj
    
    
  def checkParameters(self,obj):
    """
    """
    if obj['variable'] is None: obj['variable']=[]
    if not isinstance(obj['variable'], list):obj['variable']=[obj['variable']]
    obj["dataOnly"]=False
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
    
    obj['filepath']=os.path.join(self.apiCacheLocation,str(uuid.uuid4()))
    
    obj['groups']=groups=self.getGroups(obj)
    obj['ngroups']=ngroups=len(groups)
    obj['isTable']= ngroups==1
    return obj
  
  
  def getGroups(self,obj):
    groups=[self.getGroupsByVariable(vname) for vname in obj['variable']]
    return getUniqueGroups(groups,obj)
  
  def checkExport(self,obj):
    """
    Make sure the query is consistent with the export format
    """
    if obj['ngroups']>1 and (obj['export']=='csv' or obj['export']=='json'):raise Exception("Cannot have multiple variables without the same dimensions in a csv,json")
    if obj['export']=="geojson" and obj['ngroups']>1 and obj['variable']!=['x','y','elem','time']:raise Exception("Can only export a table or mesh to geojson")
    return obj

  
  def run(self,parameters):
    # Export Metadata Only
    variable=parameters.get('variable',None)
    if variable is None:
      meta={
        "comments":"",
        "parameters":self.getDefaultParametersExtra(),
        
      }
      return response("application/json",json.dumps(meta)) 
    
    # Check parameters
    obj=self.prepareInput(parameters)
    
    # Get data
    data=getData(self,obj)
    
    # Export data to file
    filepath=export(obj,data)
    
    # Upload to S3
    self.s3.upload(filepath)
    url=self.s3.generate_presigned_url(filepath)
    
    return responseSignedURL(url)  