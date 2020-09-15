import os
from netCDF4 import Dataset

def netcdf(data):

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