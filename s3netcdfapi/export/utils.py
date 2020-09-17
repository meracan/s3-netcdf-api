import json
import re

# helper function for csv and geojson
def get_index_list(index_string, length):
  s, e = 0, length
  ilist = list(range(s, e))
  if index_string is not None:
    if "[" in index_string:
      ilist = json.loads(index_string)
    else:
      i_i = [int(i) if i != '' else i for i in re.split(':', index_string)]
      if len(i_i) == 1:
        ilist = list(range(int(i_i[0]), int(i_i[0] + 1)))
      else:
        if i_i[0] != '': s = int(i_i[0])
        if i_i[1] != '': e = int(i_i[1])
        ilist = list(range(s, e))

  return ilist
  


    
