from .table import to_table

def to_json(obj,data):
  to_table(obj,data).to_json(obj['output']+".json")