
def checkExport(netcdf2d,obj):
  """
  Make sure the query is consistent with the export format
  """
  if obj['mesh'] and (obj['export']=='csv' or obj['export']=='json'):raise Exception("Cannot export mesh as csv or json")
  if obj['mesh'] and obj['export']=='geojson' and obj['variable'] is None:raise Exception("Cannot export mesh and variables in a geojson")
  if obj['ngroups']>1 and (obj['export']=='csv' or obj['export']=='json'or obj['export']=='geojson'):raise Exception("Cannot have multiple variables without the same dimensions in a csv,json or geojson")
  
  return obj
 