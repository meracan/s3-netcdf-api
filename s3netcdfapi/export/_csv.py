import json
from utils import get_index_list


def CSV(data):
  """

  parameter, timestep, lon, lat, value

  hs,2000-01-01T01:00:00,7.0,7.0, 71.0
  hs,2000-01-01T02:00:00,7.0,7.0, 72.0
  hs,2000-01-01T03:00:00,7.0,7.0, 73.0

  parameter,value

  lon,2.0
  lon,2.1
  lon,2.2

  parameter, timestep, lon, lat, freq, dir, value

  spectra,2000-01-01T00:00:00,16.0,16.0,0.0,0.0, 3285000.0
  spectra,2000-01-01T00:00:00,16.0,16.0,0.0,1.0, 3285001.0
  spectra,2000-01-01T00:00:00,16.0,16.0,0.0,2.0, 3285002.0

  """

  csv_string = ""
  var = data['parameter']
  t_indices = data['t_indices']
  n_indices = data['n_indices']
  #f_indices = data['f_indices'] # not used yet for now
  values = data[var]


  # other data
  if t_indices is None and n_indices is None:
    csv_string += "parameter,value\n"  # header
    for v in data[var]:
      csv_string += var + "," + str(v) + "\n"

  else:
    # get index lists
    nlist = get_index_list(n_indices, len(data['lons']))
    tlist = get_index_list(t_indices, len(data['times']))
    # flist = get_index_list(f_indices, len(data['freq'])) # not used yet for now

    if var == "spectra":
      csv_string += "parameter,timestep,lon,lat,freq,dir,value\n"  # header
      station = int(data['station'])
      f_index = data['freq']
      d_index = data['dir']
      if len(values.shape) == 2:
        values = [values.tolist()]
    else:
      csv_string += "parameter,timestep,lon,lat,value\n"  # header
      # ensure values is always a 2d list --> [[]]
      if len(values.shape) == 0:
        values = [[values.tolist()]]
      else:
        if len(tlist) == 1: values = [list(values)]
        if len(nlist) == 1: values = [[v] for v in values]

    # concatenate csv rows
    for i, t in enumerate(tlist):
      for j, n in enumerate(nlist):
        if var == "spectra":
          time = data['times'][t]
          lon = data['lats'][station][n]
          lat = data['lons'][station][n]
          for f, freq in enumerate(data['freq']):
            for d, dr in enumerate(data['dir']):
              value = values[j][f][d]
              csv_string += var + "," +\
                            str(time) + "," +\
                            str(lon) + "," +\
                            str(lat) + "," +\
                            str(freq) + "," +\
                            str(dr) + "," +\
                            str(value) +"\n"
        else:
          csv_string += var + "," +\
                        str(data['times'][t]) + "," +\
                        str(data['lons'][n]) + "," +\
                        str(data['lats'][n]) + "," +\
                        str(values[i][j]) +"\n"

  return csv_string.rstrip()
