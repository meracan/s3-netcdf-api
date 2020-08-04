import json
import binarypy
import base64
#import shapefile as shf
import numpy as np
import os
from io import BytesIO as bio
#from io import StringIO as sio
from scipy.io import loadmat, savemat
from netCDF4 import Dataset
import re
#import pprint as pp

"""
data[var] structure:

  one node
        n2
    [ [13.],  t2
      [14.],  t3
      [15.],  t4
      [16.],  t5
      [17.]   t6
    ]

  one timestep
       n2   n3   n4
    [ [3., 13., 23.]  t2
    ]

  multiple nodes/timesteps
        n2   n3   n4
    [ [ 3., 13., 23.],  t2
      [ 4., 14., 24.],  t3
      [ 5., 15., 25.],  t4
      [ 6., 16., 26.],  t5
      [ 7., 17., 27.]   t6
    ]
    
"""


def jsontest2(data):
  for var in data:
    #data[var] = list(data[var])
    data[var] = data[var].tolist() # assumes data[var] is a numpy.ndarray
  return json.dumps(data)


def saveJSON(data):
  for var in data:
    #data[var] = list(data[var])
    data[var] = data[var].tolist() # assumes data[var] is a numpy.ndarray
  return json.dumps(data)


# helper function for csv and geojson
def get_index_list(index_string, length):
  s, e = 0, length
  ilist = list(range(s, e))
  if index_string is not None:
    if "[" in index_string:
      ilist = json.loads(index_string)
    else:
      i_i = [int(i) if i != '' else i for i in re.split(':', index_string)]
      if len(i_i) == 1:
        ilist = list(range(int(i_i[0]), int(i_i[0] + 1)))
      else:
        if i_i[0] != '': s = int(i_i[0])
        if i_i[1] != '': e = int(i_i[1])
        ilist = list(range(s, e))

  return ilist


def saveGeoJSON(data):
  """
  A GeoJSON object represents a Geometry, Feature, or collection of
   Features. A GeoJSON object has a member with the name "type". The value of
   the member MUST be one of the GeoJSON types.
  Every feature has a geometry property and a properties property.

  If data[var] is multidimensional, each timestep in 'times' corresponds with an inner list in 'values'.
  Each datum in that 'values' timestep corresponds with one of the 'coordinates'.

  e.g.
  'geometry':
      { 'coordinates': [[4.0, 4.0, 4.0],
                        [5.0, 5.0, 5.0],
                        [6.0, 6.0, 6.0],
                        [7.0, 7.0, 7.0],
                        [8.0, 8.0, 8.0]],
        'type': 'Point'
      },
  'properties':
      { 'parameter': 'hs',
        'times': ['2000-01-01T05:00:00',
                  '2000-01-01T06:00:00',
                  '2000-01-01T07:00:00'],
        'values':[[54.0, 55.0, 56.0, 57.0, 58.0],
                  [64.0, 65.0, 66.0, 67.0, 68.0],
                  [74.0, 75.0, 76.0, 77.0, 78.0]]
        }


  If data[var] is one-dimensional:
  'geometry':
      { 'coordinates': [],
        'type': 'Point'
      },
  'properties':
      { 'parameter': 'lon',
        'values': [0.0,
                   1.0,
                   2.0,
                   3.0,
                   ...
                  ]
      }

  """

  var = data['parameter']
  t_indices = data['t_indices']
  n_indices = data['n_indices']
  values = data[var].tolist()

  # TODO
  if var == "spectra":
    #print("spectra data")
    pass


  # other data
  if t_indices is None and n_indices is None:
    geojson = {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "geometry": {
            "type": "Point",
            "coordinates": [],
          },
          "properties": {
            "parameter": var,
            "values": values,
          }
        }
      ]
    }

  else:
    # get index lists
    nlist = get_index_list(n_indices, len(data['lons']))
    tlist = get_index_list(t_indices, len(data['times']))

    coordinates = [
      [ data['lons'][n],
        data['lats'][n],
        data['bath'][n],
      ] for n in nlist
    ]
    times = [
      str(data['times'][t])
      for t in tlist
    ]

    geojson = {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "geometry": {
            "type": "Point",
            "coordinates": coordinates, # [ [x0, y0, z0], [x1, y1, z1], [x2, y2, z2], ...]
          },
          "properties": {
            "parameter": var,
            "values": values,
            "times": times,
          }
        }
      ]
    }

  return json.dumps(geojson)


def saveCSV(data):
  """

  parameter, timestep, lon, lat, value

  hs,2000-01-01T01:00:00,7.0,7.0,71.0
  hs,2000-01-01T02:00:00,7.0,7.0,72.0
  hs,2000-01-01T03:00:00,7.0,7.0,73.0

  parameter,value

  lon,2.0
  lon,2.1
  lon,2.2

  """

  csv_string = ""
  var = data['parameter']
  t_indices = data['t_indices']
  n_indices = data['n_indices']
  values = data[var]

  # TODO
  if var == "spectra":
    #print("spectra data")
    #values = values.tolist()
    #print(type(values), values)
    pass


  # other data
  if t_indices is None and n_indices is None:
    for v in data[var]:
      csv_string += var + "," + str(v) + "\n"

  else:
    # get index lists
    nlist = get_index_list(n_indices, len(data['lons']))
    tlist = get_index_list(t_indices, len(data['times']))

    # ensure values is always a 2d list --> [[]]
    if len(values.shape) == 0: values = [[values.tolist()]]
    else:
      if len(tlist) == 1: values = [list(values)]
      if len(nlist) == 1: values = [[v] for v in values]

    # concatenate csv rows
    for i, t in enumerate(tlist):
      for j, n in enumerate(nlist):
        csv_string += var+","+str(data['times'][t])+","+str(data['lons'][n])+","+str(data['lats'][n])+","+str(values[i][j])+"\n"

  return csv_string


def saveBinary(data):
  bpy_wdata = binarypy.write(data)
  data = base64.b64encode(bpy_wdata)
  return data
  # return binarypy.write(data)

def saveNetCDF(data):

  # TODO
  var = data['parameter']
  if var == "spectra":
    #print("spectra data")
    pass

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
  if format=="jsontest2": return response("application/json",False,jsontest2(data))

  if format=="json":return response("application/json",False,saveJSON(data))
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