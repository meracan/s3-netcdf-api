import json
import binarypy
import base64
#import shapefile as shf
import numpy as np
#import csv
import os
from io import BytesIO as bio
#from io import StringIO as sio
from scipy.io import loadmat, savemat
from netCDF4 import Dataset


def saveJSON(data):
  for var in data:
    #data[var] = list(data[var])
    data[var] = data[var].tolist() # assumes data[var] is a numpy.ndarray
  return json.dumps(data)

def saveGeoJSON(data):
  """
  A GeoJSON object represents a Geometry, Feature, or collection of
   Features. A GeoJSON object has a member with the name "type". The value of
   the member MUST be one of the GeoJSON types.
  Every feature has a geometry property and a properties property.

  Data is assumed to have the necessary information and is organized?
  """
  geojson = {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": data['coords'],  # list of coordinates
        },
        "properties": {
          "parameter": data['parameter'],
          "values": data['values']
        }
      }
    ]
  }
  return json.dumps(geojson)


def saveCSV(data):
  """
  NEEDS FIXING
  basic structure:

  "s"
        n2   n3   n4   n5   n6
    [ [ 3.,  4.,  5.,  6.,  7.],  t2
      [13., 14., 15., 16., 17.],  t3
      [23., 24., 25., 26., 27.]   t4
    ]

  "t"
        t2   t3   t4
    [ [ 3., 13., 23.],  n2
      [ 4., 14., 24.],  n3
      [ 5., 15., 25.],  n4
      [ 6., 16., 26.],  n5
      [ 7., 17., 27.]   n6
    ]

  """

  # assumes lists already
  csv_string = ""
  i = 0
  for k, var in data.items():  # only one element in the dictionary?
    csv_string += k+"\n"
    if isinstance(var[0], list) or isinstance(var[0], np.ndarray):
      for v in var:
        for vv in v:
          csv_string += str(i)+","+str(vv)+"\n"
        i += 1
    else:
      for v in var:
        csv_string += str(i)+","+str(v)+"\n"

  return csv_string


def saveBinary(data):
  bpy_wdata = binarypy.write(data)
  data = base64.b64encode(bpy_wdata)
  return data
  # return binarypy.write(data)

def saveNetCDF(data):
  # create temporary file
  with Dataset('_.nc', 'w', format='NETCDF4') as nc:
    for var in data:

      if len(data[var].shape) == 2:
        i=0
        for var2 in data[var]:
          var_name = var+"_"+str(i)
          nc.createDimension(var_name, len(var2))
          nc_var = nc.createVariable(var_name, 'f4', (var_name,))
          nc_var[:] = var2
          i+=1
          #print(nc_var[:])
      else:
        nc.createDimension(var, len(data[var]))
        nc_var = nc.createVariable(var, 'f4', (var,))
        nc_var[:] = data[var]
        #print(nc_var[:])

  # read file, then delete
  nc = Dataset('_.nc', 'r')
  os.remove('_.nc')

  return nc


def saveMat(data):
  _ = bio()
  savemat(_, {})
  mat = loadmat(_)
  for var in data:
    mat[var] = data[var]
  return mat


def saveTri(data):
  None    

def saveSLF(data):
  None

def saveShapefile(data):
  None
  
def save(format,data):
  if format=="json":return response("application/json",False,saveJSON(data))
  if format=="jsontest":return response("application/json",False,json.dumps({"test":"TestValue"}))
  if format=="geojson":return response("",False,saveGeoJSON(data))
  if format=="csv":return response("",False,saveCSV(data))
  if format=="bin":return response("application/octet-stream",True,saveBinary(data))
  if format=="nc":return response("application/octet-stream",True,saveNetCDF(data))
  if format=="mat":return response("application/octet-stream",True,saveMat(data))
  if format=="tri":return response("application/octet-stream",True,saveTri(data))
  if format=="slf":return response("application/octet-stream",True,saveSLF(data))
  if format=="shp":return response("application/octet-stream",True,saveShapefile(data))

def response(ContentType,isBase64Encoded,Body):
  return {
      'statusCode': 200,
      'headers': {"content-type": ContentType},
      'isBase64Encoded': isBase64Encoded,
      'body':Body
      # 'body': Body,
      # "headers": {"Access-Control-Allow-Origin": "*"},
    } 