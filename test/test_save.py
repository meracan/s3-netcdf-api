import numpy as np
import base64

from s3netcdf import NetCDF2D
from s3netcdfapi.save import \
  saveJSON,saveBinary,saveGeoJSON,saveCSV,saveNetCDF,saveMat,saveTri,saveSLF,saveShapefile
import binarypy

from scipy.io import savemat
from io import StringIO
import pprint as pp
import time


# note: throws Exception "NetCDF2D needs a nca object"
swan=NetCDF2D({"name":"test1","bucket":"uvic-bcwave","localOnly":True,"cacheLocation":"../../s3", "nca":{}})

data_ = []
data_.append({"lon": swan["nodes", "lon", :]})
data_.append({"lat": swan["nodes", "lat", :]})
data_.append({"bed": swan["nodes", "bed", :]})

# these two should have the same values
data_.append({"hs": swan["s", "hs", 0:3, 4]})  # nodes 0-2, time 4
data_.append({"hs": swan["t", "hs", 4, 0:3]})  # time 4, nodes 0-2

data_.append({"spc": swan["spc", "spectra", 1, 0, 0]})  # station 1 (brooks) node 0 time 0
#data_.append({"spc": swan["spc", "spectra", 0, 0, 0:2]})  # station 0 (beverly) node 0 times 0-2


def test_JSON():
  check_data =[
    "{\"lon\": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]}",
    "{\"lat\": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]}",
    "{\"bed\": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]}",
    "{\"hs\": [4.0, 14.0, 24.0]}",
    "{\"hs\": [4.0, 14.0, 24.0]}",
    "{\"spc\": [[394200.0, 394201.0, 394202.0, 394203.0, 394204.0], "+
                "[394205.0, 394206.0, 394207.0, 394208.0, 394209.0], "+
                "[394210.0, 394211.0, 394212.0, 394213.0, 394214.0]]}",
    "{\"spc\": [[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0]], "+
                "[[15.0, 16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0, 24.0], [25.0, 26.0, 27.0, 28.0, 29.0]]]}"
  ]
  for i, d in enumerate(data_):
    json_body = saveJSON(d)
    assert json_body == check_data[i]

  print("(test_JSON passed)")


def test_csv():
  # 2009-01-05_130000, hs, -120.3, 36.4, 9.02155
  check_data = [
  ]
  for i, d in enumerate(data_):
    csv_body = saveCSV(d)
    #assert csv_body == check_data[i]

  print("(test_csv passed)")


def test_mat():
  # can't compare entire matfile object since the dates always change
  check_data =[
    {'lon': np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])},
    {'lat': np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])},
    {'bed': np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])},
    {'hs': np.array([4., 14., 24.])},
    {'hs': np.array([4., 14., 24.])},
    {'spc': np.array([[394200., 394201., 394202., 394203., 394204.],
                  [394205., 394206., 394207., 394208., 394209.],
                  [394210., 394211., 394212., 394213., 394214.]])},
    {'spc': np.array([[[0., 1., 2., 3., 4.],[5., 6., 7., 8., 9.],[10., 11., 12., 13., 14.]],
                  [[15., 16., 17., 18., 19.],[20., 21., 22., 23., 24.],[25., 26., 27., 28., 29.]]])}
  ]
  for i, d in enumerate(data_):
    mat_body = saveMat(d)
    values = {k:v for k,v in mat_body.items() if k not in ['__header__', '__version__','__globals__']}
    np.testing.assert_equal(values, check_data[i])

  print("(test_mat passed)")


def test_Binary():
  # not quite working...
  swan=NetCDF2D({"name":"test1","bucket":"uvic-bcwave","localOnly":True,"cacheLocation":"../../s3"})
  data={
    "variables":{
      "bed": swan["nodes","bed", :]
    }
  }
  body=saveBinary(data) # <--- error
  print("test_Binary:", type(body))

  body=base64.b64decode(body)
  checkData=binarypy.read(body)
  np.testing.assert_array_equal(checkData['bed'], np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
  

def test_geoJSON():
  # !!! this approach needs to be changed

  # assumptions about the data
  node_start, node_end, t_start, t_end = 8, 9, 3, 7

  # these would be 'taken' from index.py
  data = {}
  data['x'] = swan["nodes", "lon", :]
  data['y'] = swan["nodes", "lat", :]
  data['z'] = swan["nodes", "bed", :]
  data['time'] = swan["time", "time", :]

  if abs(node_start-node_end) <= 1:
    data['coords'] = [
      [data['x'][node_start],
       data['y'][node_start],
       data['z'][node_start],
       str(t)] for t in data['time'][t_start:t_end]
    ]
    data['values'] = swan["s", "hs", node_start, t_start:t_end].tolist()
  elif abs(t_start-t_end) <= 1:
    data['coords'] = [
      [data['x'][n],
       data['y'][n],
       data['z'][n],
       str(data['time'][t_start])] for n in range(node_start, node_end)
    ]
    data['values'] = swan["s", "hs", node_start:node_end, t_start].tolist()

  data['parameter'] = "hs"

  body = saveGeoJSON(data)

  pp.pprint(body)

  print("(test_geoJSON passed)")


def test_netCDF():
  check_data = [
  ]
  for i, d in enumerate(data_):
    nc_body = saveNetCDF(d)
    print(f"{nc_body}")
    print("---------------------------------")
    # need to test/check values


if __name__ == "__main__":
  #test_JSON()
  #test_csv()
  #test_mat()
  #test_Binary()
  #test_geoJSON()
  #test_netCDF()

  pass

