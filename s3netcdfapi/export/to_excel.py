from .table import to_table

def to_excel(obj,data):
  filepath=obj['filepath']+".xlsx"
  to_table(obj,data).to_excel(filepath)
  return filepath