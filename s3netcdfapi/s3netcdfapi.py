import os
import boto3
import uuid
import json
import urllib
import base64
import numpy as np
import time
from s3netcdf import S3NetCDF
from mbtilesapi import getVT,readVT,send,VT2Tile

from .data import getIndex,getData
from .utils import parseParameters,getUniqueGroups,compress
from .response import response,responseSignedURL
from .export import export


def checkNetCDFExist(bucket,prefix,id,credentials):
  s3 = boto3.client('s3',**credentials)
  _prefix= "" if prefix is None else prefix+"/"
  key="{0}{1}/{1}.nca".format(_prefix,id)
  try:
    s3.head_object(Bucket=bucket, Key=key)# Check if object exist, if not returns Exception
    return True
  except Exception as e:
    raise Exception("S3NetCDF does not exist {}/{}".format(bucket,key))
    

class S3NetCDFAPI(S3NetCDF):
  # def __init__(self,*args,**kwargs):
  #   super().__init__(*args,**kwargs)
    
  
  def __enter__(self):
    super().__enter__()
    if not 'api' in self.obj['metadata']:raise Exception("S3NETCDF.nca needs S3NetCDF/metadata/api")
    if not 'dims' in self.obj['metadata']['api']:raise Exception("S3NETCDF.nca needs S3NetCDF/metadata/api/dims. Should contain dimensions,labels,types")
    if not 'spatials' in self.obj['metadata']['api']:raise Exception("S3NETCDF.nca needs S3NetCDF/metadata/api/spatials. Should contain all the spatial dimensions")
    if not 'temporals' in self.obj['metadata']['api']:raise Exception("S3NETCDF.nca needs S3NetCDF/metadata/api/temporals. Should contain all the temporal dimensions")
    if not 'spectrals' in self.obj['metadata']['api']:raise Exception("S3NETCDF.nca needs S3NetCDF/metadata/api/spectrals. Should contain all the spectral dimensions")
    self.dims=[]
    
    return self  

  def _getDim(self,key='spatials'):
    index=max(map(lambda x:self.dims.index(x) if x in self.dims else -1,getattr(self,key)))
    if index==-1:return getattr(self,key)[0]
    return getattr(self,key)[index]
  
  def _getDimObject(self,key='spatials'):
    dim=self._getDim(key)
    if not dim in self.obj['metadata']['api']:raise Exception("S3NETCDF.nca needs S3NetCDF/metadata/api/{}. Should contain all x,y properties".format(dim))
    return self.obj['metadata']['api'][dim]
  
  @property
  def spatial(self):return self._getDimObject('spatials')
  
  @property
  def temporal(self):return self._getDimObject('temporals')
  
  @property
  def spectral(self):return self._getDimObject('spectrals')

  @property
  def spatials(self):return self.obj['metadata']['api']['spatials']
  
  @property
  def temporals(self):return self.obj['metadata']['api']['temporals']
  
  @property
  def spectrals(self):return self.obj['metadata']['api']['spectrals']
  
  @property
  def alls(self):return [*self.spatials,*self.temporals,*self.spectrals]
  
  @property
  def api(self):return self.obj['metadata']['api']

  @property
  def res(self):return int(np.ceil(np.sqrt(self.obj['dimensions'][self._getDim('spatials')])))    
    
  def run(self,parameters):
      
    variable=parameters.get('variable',None)
    if variable is None:
      obj={
        "id":self.name,
        "url":"https://"+parameters.pop('url',""),
        "dimensions":self.dimensions,
        "res":self.res,
        "parameters":self.defaultParameters,
        "comments":"",
      }
      return response(json.dumps(obj),"application/json") 
    
    # Check parameters
    obj=self.prepareInput(parameters)
    
    # Get data
    start=time.time()
    data=getData(self,obj)
    # if obj['export']=="mbtiles":return export(self,obj,data)
    
    # Export data to file
    
    filepath,contentType=export(self,obj,data)
    
    # Compress data to file
    gfilepath=os.path.join(os.path.dirname(filepath),"_"+os.path.basename(filepath))
    compress(filepath,gfilepath)
    os.remove(filepath)
    
    return response(gfilepath,contentType,True)
    
    # Upload to S3
    # self.s3.upload(gfilepath,{"ContentEncoding":"gzip","ContentType":contentType})
    # url=self.s3.generate_presigned_url(gfilepath)
    # return responseSignedURL(url)    
  
  

  
  @property
  def defaultParameters(self):
    dimensions=self.dimensions
    variables=self.variables
    x=self.query({'variable':self.spatial['x']})
    y=self.query({'variable':self.spatial['y']})
    xextent=[float(np.min(x)),float(np.max(x))]
    yextent=[float(np.min(y)),float(np.max(y))]
    time=self.query({'variable':self.temporal['time']})
    
    dimensions.pop('npe',None)
    dimensions.pop('nchar',None)
    
    parameters={}
    for dname in dimensions:
      idname="i"+dname[1:]
      parameters[idname]={"default":None,"type":"int,list,slice","nvalue":dimensions[dname],"comment":"Index value of {}".format(dname)}
    
    parameters["dims"]={"default":[],"type":"str,list","comment":"Specify specific dimensions to extract data","values":self.obj['metadata']['api']['dims']}
    parameters["variable"]={"default":None,"type":"str,list","comment":"name of variables","values":list(variables.keys())}
    parameters['x']={"default":None,"type":"int,float,list","extent":xextent,"comment":"Interpolate spatial data by specifying x/y","alias":['longitude','lon','lng']}
    parameters['y']={"default":None,"type":"int,float,list","extent":yextent,"comment":"Interpolate spatial data by specifying x/y","alias":['latitude','lat']}
    parameters['station']={"default":None,"type":"str,list","comment":"Get spectra information using station/buoy name"}
    parameters["start"]={"default":None,"type":"str","comment":"Startdate (yyyy-mm-ddThh:mm:ss)","extent":[np.datetime_as_string(np.min(time), unit='s'),np.datetime_as_string(np.max(time), unit='s')]}
    parameters["end"]={"default":None,"type":"str","comment":"Endate (yyyy-mm-ddThh:mm:ss)","extent":[np.datetime_as_string(np.min(time), unit='s'),np.datetime_as_string(np.max(time), unit='s')]}
    parameters["step"]= {"default":1,"type":"int","comment":"Timestep(integer)"}
    parameters["stepUnit"]={"default":"h","type":"str","comment":"Timestep unit(s,h,d,w)"}
    parameters["inter.mesh"]={"default":"nearest","type":"str","values":["nearest","linear"],"comment":""}
    parameters["extra.mesh"]={"default":"none","type":"str","values":["none","nearest"],"comment":""}
    parameters["inter.temporal"]={"default":"nearest","type":"str","values":["nearest","linear"],"comment":"Type of spatial interpolation"}
    parameters["inter.xy"]={"default":"nearest","type":"str","values":["nearest"],"comment":"Type of spatial interpolation"}
    parameters["export"]={"default":"json","type":"str"}
    parameters["sep"]={"default":",","type":"str"}
    parameters["z"]={"default":None,"type":"int,list"}
    parameters["extra"]={"default":"true","type":"bool"}
    return parameters
  
  def prepareInput(self,parameters):
    obj=parseParameters(self.defaultParameters,parameters)
    obj=self.checkParameters(obj)
    obj=self.checkExport(obj)
    obj=getIndex(self,obj)
    return obj
  

  def checkParameters(self,obj):
    self.dims=obj.pop("dims")
    if obj['variable'] is None: obj['variable']=[]
    if not isinstance(obj['variable'], list):obj['variable']=[obj['variable']]
    obj["dataOnly"]=False
    
    if obj['export']=="slf" and not 'mesh' in obj['variable']:obj['variable'].append('mesh')
    
    if 'mesh' in obj['variable']:
      x=self.spatial['x']
      y=self.spatial['y']
      elem=self.spatial['elem']
      if not x in obj['variable']:obj['variable'].append(x)
      if not y in obj['variable']:obj['variable'].append(y)
      if not elem in obj['variable']:obj['variable'].append(elem)
      obj['variable'].remove('mesh')
      obj['inode']=None
      obj['x']=None
      obj['y']=None
      obj["dataOnly"]=True
      
    if obj['export']=="mbtiles":
      if obj['x'] is None:raise Exception("Parameter 'x' is required")
      if obj['y'] is None:raise Exception("Parameter 'y' is required")
      if obj['z'] is None:raise Exception("Parameter 'z' is required")
      obj['mx']=int(obj['x'])
      obj['my']=int(obj['y'])
      obj['mz']=int(obj['z'])
      obj['inode']=None
      obj['x']=None
      obj['y']=None
      obj['z']=None
      timeIndex="i"+self._getDim('temporals')[1:]
      itime=obj.get(timeIndex,None)
      if itime is None:raise Exception("Parameter={} needs to be added to the query".format(timeIndex))
      if not isinstance(itime,int):raise Exception("Parameter={} needs to be an integer".format(timeIndex))
      if len(obj['variable'])!=1:raise Exception("Parameter 'variable' needs only 1 variable for mbtiles")
    
    obj['filepath']=os.path.join(self.cacheLocation,"tmp",str(uuid.uuid4()))
    obj['res']=self.res;
    
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
