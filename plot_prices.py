from get_prices import get_prices
from matplotlib import pyplot as plt

def plot_prices(prices, kind, save=False, save_file_name="./stock_price.png"):
    # plot stock price
    # input;
    # prices
    # kind
    # save_path

    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1,1,1)

    plt.title(kind + " stock prices")
    plt.xlabel("day")
    plt.ylabel("price")
    ax.plot(prices)

    plt.show()
    if save:
        plt.savefig(save_file_name)

def get_plot_price(code, start, end=None, interval="d", kind="Open"):
    try:
        price = get_prices(code, start="2016/05/01", end=end, interval="m", kind=kind, verbose=True, output_dataframe=True)
    except:
        print("get price error")

    plot_prices(price, kind=kind)

if __name__ == "__main__":
    get_plot_price(9104, "2016/06/01", interval='d', kind="Open")
