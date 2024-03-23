import matplotlib.pyplot as plt
import seaborn as sea

sea.set_style("whitegrid")
sea.set_context("notebook")

def Plot(df, tckr, columns, title):
    for col in columns:
        plt.plot(df[tckr][col])
        plt.title(title)
        plt.show()
        
        
def Plot_Predictions(df, tckr):
    fig, axs = plt.subplots(2, 2)
    cols = df[tckr].columns
    for i in range(4):
        sea.lineplot(df[tckr][cols[2*i:2*i+2]], ax=axs[i%2][i//2])
    fig.suptitle(tckr)
    plt.tight_layout()
    plt.show()