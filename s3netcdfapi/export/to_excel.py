from .table import to_table

def to_excel(obj,data):
  to_table(obj,data).to_excel(obj['output']+".xlsx")