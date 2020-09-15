import export

def _export(format,data):
  """
  Parameters
  ----------
  format:str,choice(json,geojson,...)
  data:ndarray
  
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
  
  if format=="jsontest": return export.jsontest(data)
  if format=="json":return export.json(data)
  if format=="geojson":return export.geojson(data)
  if format=="csv":return export.csv(data)
  if format=="bin":return export.binary(data)
  if format=="nc":return export.netcdf(data)
  if format=="mat":return export.mat(data)
  if format=="tri":return export.tri(data)
  if format=="slf":return export.slf(data)
  if format=="shp":return export.shapefile(data)

