import matplotlib.pyplot as plt

def Plot(df, tckr, columns):
    for col in columns:
        plt.plot(df[tckr][col])