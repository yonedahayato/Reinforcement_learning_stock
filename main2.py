# -*- coding: utf-8 -*-

class DecisiopnPolicy:
    def select_action(self, current_state):
        pass
    def update_q(self, state, action, reward, next_state):
        pass

class RandomDesiopnPolicy(DecisiopnPolicy):
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, current_state):
        action = self.actions[random.randint(0, len(self.actions)-1)]
        return action
