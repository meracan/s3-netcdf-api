import export

def _export(parameters,netcdf2d):
  """
  Parameters
  ----------
  parameters:
  netcdf2d:ndarray

  """
  format = parameters.pop('format',"json")
  if format=="jsontest": return ("application/json",False,export.jsontest(parameters,netcdf2d))
  if format=="json":return ("application/json",False,export.json(parameters,netcdf2d))
  if format=="geojson":return ("application/json",False,export.geojson(parameters,netcdf2d))
  if format=="csv":return ("text/csv",False,export.csv(parameters,netcdf2d))
  if format=="bin":return ("application/octet-stream",True,export.binary(parameters,netcdf2d))
  if format=="nc":return ("application/octet-stream",True,export.netcdf(parameters,netcdf2d))
  if format=="mat":return ("application/octet-stream",True,export.mat(parameters,netcdf2d))
  if format=="tri":return ("application/octet-stream",True,export.tri(parameters,netcdf2d))
  if format=="slf":return ("application/octet-stream",True,export.slf(parameters,netcdf2d))
  if format=="shp":return ("application/octet-stream",True,export.shapefile(parameters,netcdf2d))

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