import json
from utils import get_index_list

def geojson(data):
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

  For var == "spectra":
  'geometry':
      { 'coordinates': [[8.0, 8.0],
                        [16.0, 16.0],
        'type': 'Point'
      },
  'properties':
      { 'parameter': 'spectra',

        'times': ['2000-01-01T05:00:00'],

        'freq': [0.0, 1.0, 2.0],
        'dir':  [0.0, 1.0, 2.0, 3.0, 4.0],

        'values':[
                  [[90.0, 91.0, 92.0, 93.0, 94.0],
                  [95.0, 96.0, 97.0, 98.0, 99.0],
                  [100.0, 101.0, 102.0, 103.0, 104.0]],

                  [[240.0, 241.0, 242.0, 243.0, 244.0],
                  [245.0, 246.0, 247.0, 248.0, 249.0],
                  [250.0, 251.0, 252.0, 253.0, 254.0]
                ]
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
  coordinates, properties = [], {}

  # other data
  if t_indices is None and n_indices is None:
    properties = {
      "parameter": var,
      "values": values,
    }

  else:
    # get index lists
    nlist = get_index_list(n_indices, len(data['lons']))
    tlist = get_index_list(t_indices, len(data['times']))

    times = [
      str(data['times'][t])
      for t in tlist
    ]

    if var == "spectra":
      station = int(data['station'])
      freq = list(data['freq'])
      dir_ = list(data['dir'])

      coordinates = [
        [data['lons'][station][n],
         data['lats'][station][n]
         ] for n in nlist
      ]

      properties = {
        "parameter": var,
        "values": values,
        "times": times,
        "frequencies": freq,
        "directions": dir_
      }

    else:

      coordinates = [
        [data['lons'][n],
         data['lats'][n],
         data['bath'][n],
         ] for n in nlist
      ]

      properties = {
        "parameter": var,
        "values": values,
        "times": times,
      }

  # [ [x0, y0, z0], [x1, y1, z1], [x2, y2, z2], ...]
  geojson = {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": coordinates,
        },
        "properties": properties
      }
    ]
  }

  #pp.pprint(values)

  return json.dumps(geojson)
