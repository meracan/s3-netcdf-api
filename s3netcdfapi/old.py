# from utils import parseParameter
# from interpolation import sInterpolate
# def main(parameters,netcdf2d):
#   """ 
#   array-string = "0,1,2"=>[0,1,2]
  
#   parameters:
#     export:string
#     mesh:true or false
#     variable:string,array-string (b,u)
#     frame:number,array-string,slice
#     node:number,array-string,slice
#     start:string (format=yyyy-mm-dd-hh)
#     end:string (format=yyyy-mm-dd-hh)
#     time:string,array-string
#     longitude:number,array-string
#     latitude:number,array-string
#     method:(closest or interpolated)
  
#   Scenarios
#   mesh:
#     Ignores node,longitude,latitude,method
#     Mesh is ignore for Table Format
  
#   variable + table format:
#     Variables needs to be in the same group
  
  
#   geojson: mesh==True=>mesh only =>Polygon
  
#   Table
#     property (x,y,time,u,v)
    
#     json 
#     geojson points
#     csv
  
#   tri
#     mesh (nodes,connectibity)
#     u
#     v
  
#   slf
#     mesh
  
#   mat
  
#   netcdf
#     mesh=>array
#   """
  
  
#   obj={
#     'export':"json",
#     'mesh':"false",
#     'variable':None,
#     'inode':None,
#     'longitude':None,
#     'latitude':None,
#     'x':None,
#     'y':None,
#     'itime':None,
#     'start':None,
#     'end':None,
#     'step':None,
#     'smethod':'closest',
#     'tmethod':'closest',
#   }
#   for o in obj:
#     obj[o]=parseParameter(parameters.get(o,obj[o]))
  
#   obj=checkParameters(obj,netcdf2d)
#   mesh=netcdf2d.getMesh()

    
 
    
    
# def getSpectra():
#   None
#   # if variable.contains("spectra"):
#       # if node:
#         # get node
#       # elif x and y:
#         # get nearest node
    
# # Array to Table ((array,1d,2d),index=(None,"",[""]))

# # get-variable frame=0:0 node=0:0 time='2020-01-12:2020-01-02' lng=1 lat=1
# # post-variable 
# # frame:0:0,[]
# # node=0:0,[]
# # (name of variable)time="",[]
# # (name of variable)x=0,[]
# # (name of variable)y=0,[]
# # (name of variable)u
# # method=closest, interpolated
# # export
# #

# # variable="(mesh+frame) (x,y,timeseries) (x,y,frame)" frame=0 node=0 x=0

# #
# # binary, 
# # mat
# # netcdf
# # shapefile
# # slf
# # tri

 

# def linearTriInterpolator(elem,x,y):

  


  
  

# def linear(netcdf2d,obj,variable):
#   """
#   """

  
#   # tmpData=netcdf2d.query(cleanObject({**obj,'variable':variable,'inode':meshId}))
#   # if len(obj['itime'])!=tmpData.shape[0]:
#     # tmpData=tmpData.T
#   # tmpData=np.squeeze(tmpData)
  
  
#   if tmpData.ndim==1:
#     z=np.zeros(nnode)
#     z[meshId]=tmpData
#     lti=LinearTriInterpolator(tri,z,trifinder)
#     return lti(obj['x'], obj['y'])




# def cleanObject(obj):
#   newobject={}
  
#   indim=obj['pointer']['indim']
#   for name in indim:
#     if not name in obj:raise Exception("Add {} to the default parameter object".format(name))
#     dim=obj.get(name)
#     newobject[indim[name]]=dim
  
#   if not 'variable' in obj:raise Exception("Add {} to the default parameter object".format('variable'))
#   newobject['variable']=obj['variable']
  
#   return newobject
  
# def sInterpolate(netcdf2d,obj,variable):
#   """
#   """
#   # if len(obj['itime'])!=1:raise Exception("There should be only 1 time step")
#   if obj['smethod']=='idw':return IDW(netcdf2d,obj,variable)
#   elif obj['smethod']=='linear':return linear(netcdf2d,obj,variable)
#   else: return closest(netcdf2d,obj,variable)

# def IDW(netcdf2d,obj,variable,regularize_by=1e-9):
#   """
#   """  
  
#   obj=getMesh(netcdf2d,obj)
#   xy=np.column_stack((obj['meshx'],obj['meshy']))
 
#   kdtree = cKDTree(xy)
#   distances,ids=kdtree.query(obj['xy'],3)
#   distances += regularize_by
#   meshId,meshId2ids=np.unique(ids.ravel(),return_inverse=True)
#   tmpData=netcdf2d.query(cleanObject({**obj,'variable':variable,'inode':meshId}))
  
#   tmpData=np.squeeze(tmpData)
#   tmpData=tmpData[meshId2ids]
#   weights = tmpData.reshape(ids.shape)
#   return np.sum(weights/distances, axis=1) / np.sum(1./distances, axis=1)

    

# # def linear(netcdf2d,obj,variable):
# #   """
# #   """
# #   obj=getMesh(netcdf2d,obj)
# #   nnode=len(obj['meshx'])
# #   tri = Triangulation(obj['meshx'], obj['meshy'], obj['elem'].astype("int32"))
# #   trifinder = tri.get_trifinder()
# #   ids=obj['elem'][trifinder.__call__(obj['x'], obj['y'])].astype("int32")
# #   meshId=np.unique(ids.ravel())
# #   tmpData=netcdf2d.query(cleanObject({**obj,'variable':variable,'inode':meshId}))
# #   if len(obj['itime'])!=tmpData.shape[0]:
# #     tmpData=tmpData.T
# #   tmpData=np.squeeze(tmpData)
  
  
# #   if tmpData.ndim==1:
# #     z=np.zeros(nnode)
# #     z[meshId]=tmpData
# #     lti=LinearTriInterpolator(tri,z,trifinder)
# #     return lti(obj['x'], obj['y'])

 

# def closest(netcdf2d,obj,variable):
#   """
#   """
#   obj=getMesh(netcdf2d,obj)
#   xy=np.column_stack((obj['meshx'],obj['meshy']))
#   kdtree = cKDTree(xy)
#   distance,id=kdtree.query(obj['xy'],1)
#   tmpData=netcdf2d.query(cleanObject({**obj,'variable':variable,'inode':id}))
#   if "temporal-spatial":None
#   if "spatial":None
#   if "spectral":None
  
  
    
#   if obj['itime'] is not None and len(obj['itime'])!=tmpData.shape[0]:
#     tmpData=tmpData.T

    
#   return tmpData

# def tInterpolate(netcdf2d,obj,variable):
#   """
#   """
#   if obj['tmethod']=='closest':return tclosest(netcdf2d,obj,variable)
#   else: return tlinear(netcdf2d,obj,variable)

# def tlinear(netcdf2d,obj,variable):
#   """
#   """  
#   None
# def tclosest(netcdf2d,obj,variable):
#   """
#   """  
#   None  

  
