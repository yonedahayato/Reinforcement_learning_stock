from policys import RandomDesiopnPolicy
from get_prices import get_prices
from sub_module import sub_run
import sys
import numpy as np

def run_simulation(policy, initial_budget, initial_num_stock, prices, hist, debug=False, verbose=False):
    portfolios = []
    budget = initial_budget
    portfolios.append(budget)

    num_stocks = initial_num_stock
    share_value = 0
    transitions = list()

    if verbose:
        print("budget: {}".format(initial_budget))

    for i in range(len(prices) - hist - 1):
        current_state = np.asmatrix(np.hstack((prices[i:i+hist], budget, num_stocks))) # 現時点の状況

        current_portfolio = budget + num_stocks * share_value # 現時点の総額

        action = policy.select_action(current_state, i) # 次に行うアクション

        share_value = float(prices[i + hist + 1]) # 次に行うアクション(buy or sell)で取引される株価

        if action == "Buy" and budget >= share_value:
            budget -= share_value
            num_stocks += 1
        elif action == "Sell" and num_stocks > 0:
            budget += share_value
            num_stocks -= 1
        else:
            action = "Hold"

        new_portfolio = budget + num_stocks * share_value
        portfolios.append(new_portfolio)

        reward = new_portfolio - current_portfolio

        next_state = np.asmatrix(np.hstack((prices[i+1:i+hist+1], budget, num_stocks)))

        transitions.append((current_state, action, reward, next_state))

        policy.update_q(current_state, action, reward, next_state)

        if verbose:
            print("{}: action is {}".format(i, action))
            print("new budget: {}".format(new_portfolio))

    portfolio = budget + num_stocks * share_value
    if debug:
        print("${}¥t{} shares".format(budget, num_stocks))

    sub_run.plot_stock_price_change(portfolios, save_path="./picture/stock_price_change/")

    return portfolio

def run_simulations(policy, budget, num_stocks, prices, hist):
    print("prices_num: {}, hist_num: {}".format(len(prices), hist))
    for i in range(5):
        checker = input("check these numbers -> [y/n] :")
        if checker == "n":
            print("ok, stop program")
            sys.exit()
        elif checker == "y":
            print("ok, i will proced")
            break
        else:
            print("please input 'y' or 'n'")
            continue

    num_tries = 10
    final_portfolios = list()

    for i in range(num_tries):
        final_portfolio = run_simulation(policy, budget, num_stocks, prices, hist)
        final_portfolios.append(final_portfolio)
        print("final budget: {}".format(final_portfolio))

    avg, std = np.mean(final_portfolios), np.std(final_portfolios)

    return avg, std

if __name__ == "__main__":
    prices = get_prices(9104, start="2015/01/01", end="2016/01/01", interval="d", kind="Open",
                        verbose=False, output_dataframe=False, length_info=False)
    #plot_prices(prices)

    actions = ["Buy", "Sell", "Hold"]
    hist = 20

    policy = RandomDesiopnPolicy(actions)
    budget = 1000.0
    num_stocks = 0

    avg, std = run_simulations(policy, budget, num_stocks, prices, hist)
    print("avg: {}, std: {}".format(avg, std))
