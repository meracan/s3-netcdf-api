import json

def jsontest(data):
  for var in data:
    #data[var] = list(data[var])
    data[var] = data[var].tolist() # assumes data[var] is a numpy.ndarray
  return json.dumps(data)