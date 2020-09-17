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

 
