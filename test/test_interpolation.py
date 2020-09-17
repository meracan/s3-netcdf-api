
import s3netcdfapi.interpolation as inter

def test_IDW():
    inter.IDW()
    
def test_linear():
    inter.linear()
    

def test_closest():
    inter.closest()
    

def test_tlinear():
    inter.tlinear()
    

def test_tclosest():
    inter.tclosest()
    

if __name__ == "__main__":
  test_IDW()
  test_linear()
  test_closest()
  test_tlinear()
  test_tclosest()