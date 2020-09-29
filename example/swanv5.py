import pandas as pd




def matplotlib_example():
    import matplotlib.pyplot as plt
    df = pd.read_csv("https://w4wj62194j.execute-api.us-west-2.amazonaws.com/Prod?variable=hs&inode=0&export=csv")
    plt.plot_date(df['Datetime'].astype("datetime64[h]"),df['hs,m'],"-")
    plt.savefig("hs_0.matplotlib.png")

def plotly_example():
    import plotly.graph_objects as go
    df = pd.read_csv("https://w4wj62194j.execute-api.us-west-2.amazonaws.com/Prod?variable=hs&inode=0&export=csv")
    fig = go.Figure([go.Scatter(x=df['Datetime'], y=df['hs,m'])])
    fig.write_image("hs_0.plotly.png")
    
    
if __name__ == "__main__":
  matplotlib_example()
  plotly_example()
      