# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random, sys, os

import tensorflow as tf

def get_quote(code, start=None, end=None, interval='d', kind="Open"):
    base = 'http://info.finance.yahoo.co.jp/history/?code={0}.T&{1}&{2}&tm={3}&p={4}'

    start = pd.to_datetime(start)
    if end == None:
        end = pd.to_datetime(pd.datetime.now())
    else :
        end = pd.to_datetime(end)

    start = 'sy={0}&sm={1}&sd={2}'.format(start.year, start.month, start.day)
    end = 'ey={0}&em={1}&ed={2}'.format(end.year, end.month, end.day)

    if interval not in ['d', 'w', 'm', 'v']:
        raise ValueError("Invalid interval: valid values are 'd', 'w', 'm' and 'v'")

    p = 1
    results = []
    while True:
        url = base.format(code, start, end, interval, p)
        tables = pd.read_html(url, header=0)
        if len(tables) < 2 or len(tables[1]) == 0: break
        results.append(tables[1])
        p += 1

    result = pd.concat(results, ignore_index=True)

    result.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    if interval == 'm':
        result['Date'] = pd.to_datetime(result['Date'], format='%Y年%m月')
    else:
        result['Date'] = pd.to_datetime(result['Date'], format='%Y年%m月%d日')
    result = result.set_index('Date')
    result = result.sort_index()
    result = list(result.ix[0:, kind])

    return result

def plot_prices(prices):
    plt.title("Opening stock prices")
    plt.xlabel("day")
    plt.ylabel("price")
    plt.plot(prices)

    plt.show()
    #plt.savefig("prices.png")

class DecisiopnPolicy:
    def select_action(self, current_state, time):
        pass
    def update_q(self, state, action, reward, next_state):
        pass

class RandomDesiopnPolicy(DecisiopnPolicy):
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, current_state, time):
        action = self.actions[random.randint(0, len(self.actions)-1)]
        return action

class QLearningDecisionPolicy(DecisiopnPolicy):
    def __init__(self, actions, input_dim):
        self.epsilon = 0.9
        self.gamma = 0.01
        self.actions = actions
        output_dim = len(actions)
        h1_dim = 200

        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y = tf.placeholder(tf.float32, [output_dim])

        w1 = tf.Variable(tf.random_normal([input_dim, h1_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h1 = tf.nn.relu(tf.matmul(self.x, w1) + b1)

        w2 = tf.Variable(tf.random_normal([h_dim, output_dim]))
        b2 = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        self.q = tf.nn.relu(tf.matmul(h1, w2) + b2)

        loss = tf.square(self.y - self.q)
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def select_action(self, current_state, step):
        threshold = min(self.epsilon, step / 1000.)

        if random.random() < threshold:
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            action_idx = np.argmax(action_q_vals)
            action = self.actions[action_idx]

        else:
            action = self.actions[random.randint(0, len(self.actions) - 1)]

        return action

    def update_q(self, state, action, reward, next_state):
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})

        next_action_idx = np.argmax(next_action_q_vals)
        action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})


def run_simulation(policy, initial_budget, initial_num_stock, prices, hist, debug=False):
    budget = initial_budget
    num_stocks = initial_num_stock
    share_value = 0
    transitions = list()
    for i in range(len(prices) - hist - 1):
        if i % 100:
            print("progress {:.2f}%".format(float(100*i) / (len(prices) - hist -1)))

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

        reward = new_portfolio - current_portfolio

        next_state = np.asmatrix(np.hstack((prices[i+1:i+hist+1], budget, num_stocks)))

        transitions.append((current_state, action, reward, next_state))

        policy.update_q(current_state, action, reward, next_state)

    portfolio = budget + num_stocks * share_value
    if debug:
        print("${}¥t{} shares".format(budget, num_stocks))

    return portfolio

def run_simulations(policy, budget, num_stocks, prices, hist):
    num_tries = 10
    final_portfolios = list()

    for i in range(num_tries):
        final_portfolio = run_simulation(policy, budget, num_stocks, prices, hist)
        final_portfolios.append(final_portfolio)

    avg, std = np.mean(final_portfolios), np.std(final_portfolios)

    return avg, std

if __name__ == "__main__":
    prices = get_quote(9104, start="2015/05/01", kind="Open")
    plot_prices(prices)

    actions = ["Buy", "Sell", "Hold"]
    hist = 200
    policy = RandomDesiopnPolicy(actions)
    budget = 1000.0
    num_stocks = 0
    avg, std = run_simulations(policy, budget, num_stocks, prices, hist)
    print(avg, std)
