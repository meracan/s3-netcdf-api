from .table import to_table
from .to_csv import to_csv
from .to_excel import to_excel
from .to_json import to_json
from .to_geojson import to_geojson
from .netcdf import to_netcdf
from .slf import to_slf


def export(obj,data):
  format = obj.pop('export',"json")
  if format=="json":return to_json(obj,data)
  if format=="geojson":return to_geojson(obj,data)
  if format=="csv":return to_csv(obj,data)
#   if format=="bin":return to_binary(obj,data)
  if format=="netcdf":return  to_netcdf(obj,data)
#   if format=="mat":return  to_csv(obj,data)
#   if format=="tri":return  to_csv(obj,data)
#   if format=="slf":return  to_slf(obj,data)
#   if format=="shp":return  to_csv(obj,data)

