import random
import tensorflow as tf

class DecisiopnPolicy:
    def select_action(self, current_state, time):
        # given a state, the decision policy will calculate the next action to take
        # input;
        # current_state,,,
        # time

        # output;

        pass
    def update_q(self, state, action, reward, next_state):
        # input;
        # state
        # action
        # reward
        # next_state

        # output;

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
