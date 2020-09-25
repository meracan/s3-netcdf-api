from .table import to_table

def to_json(obj,data):
  filepath=obj['filepath']+".json"
  to_table(obj,data).to_json(filepath)
  return filepath