from .table import to_table
from .to_csv import to_csv
from .to_excel import to_excel
from .to_json import to_json
from .to_geojson import to_geojson
from .to_netcdf import to_netcdf
from .to_binary import to_binary
from .to_mat import to_mat

from .slf import to_slf


def export(netcdf2d,obj,data):
  format = obj.pop('export',"json")
  if format=="json":return to_json(obj,data),"text/json"
  if format=="geojson":return to_geojson(obj,data),"text/json"
  if format=="csv":return to_csv(obj,data),"test/csv"
  if format=="bin":return to_binary(netcdf2d,obj,data),"application/bin"
  if format=="netcdf":return  to_netcdf(obj,data),"application/nc"
  if format=="mat":return  to_mat(obj,data),"application/mat"
#   if format=="tri":return  to_csv(obj,data),"application/tri"
#   if format=="slf":return  to_slf(obj,data),"application/slf"
#   if format=="shp":return  to_csv(obj,data),"application/shp"

