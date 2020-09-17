from s3netcdfapi import handler,query,getSpatial,getTemporal

def test_handler():
    handler()
    
def test_query():
    query()

def test_getSpatial():
    getSpatial()

def test_getTemporal():
    getTemporal()


if __name__ == "__main__":
  test_handler()
  test_query()
  test_getSpatial()
  test_getTemporal()