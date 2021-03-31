import os
import boto3
import uuid
import json
import urllib
import base64
import numpy as np
from s3netcdf import NetCDF2D
from .data import getIndex,getData
from .utils import parseParameters,getUniqueGroups,compress
from .response import response,responseSignedURL
from .export import export

class S3NetCDFAPI(NetCDF2D):
  def __init__(self, obj):
    super().__init__(obj)
    
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
    id = parameters.get("id",os.environ.get("AWS_DEFAULTMODEL",None))
    if id is None:raise Exception("Api needs a model id")
    
    isDebug=os.environ.get('AWS_DEBUG',"True")
    if isDebug=="True":
      bucket=os.environ.get('AWS_BUCKETNAME',"uvic-bcwave")
      bucket=parameters.get('bucket',bucket)
      prefix = os.environ.get("AWS_PREFIX",None)
      localOnly = parameters.get("localOnly",True)
      netcdf2d=S3NetCDFAPI({"name":id,"s3prefix":prefix,"bucket":bucket,"verbose":True,"localOnly":localOnly,"cacheLocation":r"../s3","apiCacheLocation":r"../s3/tmp","credentials":credentials})
    else:
   
      bucket=os.environ.get('AWS_BUCKETNAME',None)
      prefix = os.environ.get("AWS_PREFIX",None)
      cache=os.environ.get('AWS_CACHE','/tmp')
      cachePath=os.path.join(cache,"tmp")
      if not os.path.exists(cachePath):os.makedirs(cachePath)
      S3NetCDFAPI.checkNetCDFExist(credentials,id,prefix,bucket)
      netcdf2d=S3NetCDFAPI({"name":id,"prefix":prefix,"bucket":bucket,"localOnly":False,"cacheLocation":cache,"apiCacheLocation":cache+"/tmp"})
    return netcdf2d

  def run(self,parameters):
    # Export Metadata Only
    url=parameters.pop('url',"")
    variable=parameters.get('variable',None)
    if variable is None:
      meta={
        "id":self.name,
        "url":"https://"+url,
        "dimensions":self._meta['dimensions'],
        "res":self.res(),
        "variables":self.getVariables(),
        "parameters":self.getDefaultParametersExtra(),
        "comments":"",
      }
      return response("application/json",json.dumps(meta)) 
    
    # Check parameters
    obj=self.prepareInput(parameters)
    
    # Get data
    data=getData(self,obj)
    
    # Export data to file
    filepath,contentType=export(self,obj,data)
    
    # Compress data to file
    gfilepath=os.path.join(os.path.dirname(filepath),"_"+os.path.basename(filepath))
    compress(filepath,gfilepath)
    
    # Upload to S3
    self.s3.upload(gfilepath,{"ContentEncoding":"gzip","ContentType":contentType})
    url=self.s3.generate_presigned_url(gfilepath)
    
    return responseSignedURL(url)    
    
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
    
    parameters["variable"]={"default":None,"type":(str,list),"comment":"name of variables"}
    
    parameters['x']={"default":None,"type":(float,list),"comment":"Interpolate spatial data by specifying x/y"}
    parameters['y']={"default":None,"type":(float,list),"comment":"Interpolate spatial data by specifying x/y"}
    
    parameters['longitude']={"default":None,"type":(float,list),"comment":"Interpolate spatial data by specifying longitude/latitude"}
    parameters['latitude']={"default":None,"type":(float,list),"comment":"Interpolate spatial data by specifying longitude/latitude"}
    parameters['lon']={"default":None,"type":(float,list),"comment":"Interpolate spatial data by specifying lon/lat"}
    parameters['lat']={"default":None,"type":(float,list),"comment":"Interpolate spatial data by specifying lon/lat"} 
    parameters['sx']={"default":None,"type":(float,list),"comment":"Interpolate spatial/spectral data by specifying"}
    parameters['sy']={"default":None,"type":(float,list),"comment":"Interpolate spatial/spectral data by specifying x/y"}
    parameters['slon']={"default":None,"type":(float,list),"comment":"Interpolate spatial/spectral data by specifying lon/lat"}
    parameters['slat']={"default":None,"type":(float,list),"comment":"Interpolate spatial/spectral data by specifying lon/lat"}
    parameters['station']={"default":None,"type":(str,list),"comment":"Get spectra information using station/buoy name"}
    
    parameters["start"]={"default":None,"type":str,"comment":"Startdate (yyyy-mm-ddThh:mm:ss)"}
    parameters["end"]={"default":None,"type":str,"comment":"Endate (yyyy-mm-ddThh:mm:ss)"}
    parameters["step"]= {"default":1,"type":int,"comment":"Timestep(integer)"}
    parameters["stepUnit"]={"default":"h","type":str,"comment":"Timestep unit(s,h,d,w)"}
    parameters["inter.mesh"]={"default":"nearest","type":str,"values":["nearest","linear"],"comment":""}
    parameters["extra.mesh"]={"default":"none","type":str,"values":["none","nearest"],"comment":""}
    parameters["inter.temporal"]={"default":"nearest","type":str,"values":["nearest","linear"],"comment":"Type of spatial interpolation"}
    parameters["inter.xy"]={"default":"nearest","type":str,"values":["nearest"],"comment":"Type of spatial interpolation"}
    parameters["export"]={"default":"json","type":str}
    parameters["sep"]={"default":",","type":str}
  
    return parameters
  
  @property
  def spatial(self):return self._meta['metadata'].get('spatial',{"x":"x","y":"y","elem":"elem","dim":"nnode"})
  
  @property
  def temporal(self):return self._meta['metadata'].get('temporal',{"time":"time","dim":"ntime"})
  
  @property
  def spectral(self):return self._meta['metadata'].get('spectral',{"sx":"sx","sy":"sy","dim":"nsnode","stationId":"stationid","stationName":"name"})

  def getDefaultParametersExtra(self):
    parameters=self.getDefaultParameters()
    
    variables=self.getVariables()
    vnames=list(variables.keys())    
    parameters['variable']=vnames
    
    xname=self.spatial['x']
    yname=self.spatial['y']
    x=self.query({'variable':xname})
    y=self.query({'variable':yname})
    parameters[xname]={'extent':[float(np.min(x)),float(np.max(x))]}
    parameters[yname]={'extent':[float(np.min(y)),float(np.max(y))]}
    
    
    # TODO: only for BCSWAN
    # sxname=self.spectral['sx']
    # syname=self.spectral['sy']
    # sx=self.query({'variable':sxname})
    # sy=self.query({'variable':syname})
    # parameters[sxname]={'extent':[float(np.min(sx)),float(np.max(sx))]}
    # parameters[syname]={'extent':[float(np.min(sy)),float(np.max(sy))]}    
    
    timename=self.temporal['time']
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
    return {**parseParameters(self.getDefaultParameters(),parameters)}
  
  def res(self):
    return int(np.ceil(np.sqrt(self._meta['dimensions'][self.spatial['dim']])))
  
  def prepareInput(self,parameters):
    """
    """
    obj=self.getParameters(parameters)
    obj=self.checkParameters(obj)
    obj=self.checkExport(obj)
    obj=getIndex(self,obj)
    return obj
  
  # def getCacheName(self,obj):
  #   obj = sorted(obj.items(), key=lambda val: val[0])
    
  #   params = urllib.parse.urlencode(obj)
    
  #   # encodedBytes = base64.b64encode(params.encode("utf-8"))
  #   # encodedStr = str(encodedBytes, "utf-8")/
  #   print(len(params))
  #   # print(encodedStr)
  #   # print(params)
    
  def checkParameters(self,obj):
    """
    """
    if obj['variable'] is None: obj['variable']=[]
    if not isinstance(obj['variable'], list):obj['variable']=[obj['variable']]
    obj["dataOnly"]=False
    
    if obj['longitude'] is not None:obj['x']=obj['longitude'];del obj['longitude']
    if obj['latitude'] is not None:obj['y']=obj['latitude'];del obj['latitude']
    if obj['lon'] is not None:obj['x']=obj['lon'];del obj['lon']
    if obj['lat'] is not None:obj['y']=obj['lat'];del obj['lat']  
    
    if obj['sx'] is not None:obj['x']=obj['sx'];del obj['sx']
    if obj['sy'] is not None:obj['y']=obj['sy'];del obj['sy']
    if obj['slon'] is not None:obj['x']=obj['slon'];del obj['slon']
    if obj['slat'] is not None:obj['y']=obj['slat'];del obj['slat']
    
    if obj['export']=="slf" and not 'mesh' in obj['variable']:
      obj['variable'].append('mesh')
    if 'mesh' in obj['variable']:
      x=self.spatial['x']
      y=self.spatial['y']
      elem=self.spatial['elem']
      if not x in obj['variable']:obj['variable'].append(x)
      if not y in obj['variable']:obj['variable'].append(y)
      if not elem in obj['variable']:obj['variable'].append(elem)
      obj['variable'].remove('mesh')
      obj['inode']=None
      obj['x']=None;obj['y']=None
      obj['longitude']=None;obj['latitude']=None
      obj['lon']=None;obj['lat']=None      
      obj["dataOnly"]=True
    
    obj['filepath']=os.path.join(self.apiCacheLocation,str(uuid.uuid4()))
    obj['res']=self.res();
    
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

  
