
def checkExport(netcdf2d,obj):
  """
  Make sure the query is consistent with the export format
  """
  if obj['ngroups']>1 and (obj['export']=='csv' or obj['export']=='json'):raise Exception("Cannot have multiple variables without the same dimensions in a csv,json")
  if obj['export']=="geojson" and obj['ngroups']>1 and obj['variable']!=['x','y','elem']:raise Exception("Can only export a table or mesh to geojson")
  
  
  
  return obj
 