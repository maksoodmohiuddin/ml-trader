"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=1000, \
        num_actions = 3, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        # initialize Q table
        self.Q = np.random.uniform(low=-0.001, high=0.001, size=(num_states, num_actions))
        #self.Q = np.zeros((num_states,num_actions))

        #print self.Q

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new states
        @returns: The selected action
        """

        # roll the dice to decide whether to take random action or not
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q[s,:])

        # update s & a for next
        self.s = s
        self.a = action

        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        # update Q table
        # The formula for computing Q for any state-action pair <s, a>, given an experience tuple <s, a, s', r>, is:
        # Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * Q[s', argmax a'(Q[s', a'])])
        # r = R[s, a] is the immediate reward for taking action a in state s,
        # (gamma) is the discount factor used to progressively reduce the value of future rewards,
        # s' is the resulting next state,
        # argmax a'(Q[s', a']) is the action that maximizes the Q-value among all possible actions a' from s', and,
        # (alpha) is the learning rate used to vary the weight given to new experiences compared with past Q-values.

        s_prime_best_action = np.argmax(self.Q[s_prime,:])

        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (r + self.gamma * self.Q[s_prime, s_prime_best_action])

        # roll the dice to decide whether to take random action or not
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = s_prime_best_action

        # decay rar
        self.rar = self.rar * self.radr

        # save s & a for next update
        self.a = action
        self.s = s_prime

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"