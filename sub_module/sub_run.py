import matplotlib.pyplot as plt
from datetime import datetime

def plot_stock_price_change(portfolios, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(len(portfolios)), portfolios)
    if save_path == None:
        plt.show()
    elif save_path != None:
        today = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        plt.savefig(save_path+today+".png")
