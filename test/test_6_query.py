from s3netcdfapi.index import query


def test_query():
    
    # query({"id":"input1","export":"csv","variable":"u,v","inode":[0,1,2],"itime":[0,1]})
    query({"id":"input1","export":"csv","variable":"u,v","x":-160.0,"y":40.0,"itime":[0,1]})
    
    

if __name__ == "__main__":
  test_query()
  