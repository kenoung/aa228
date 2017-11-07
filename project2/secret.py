import logging

import numpy as np
from tqdm import tqdm


class Secret(object):
    def __init__(self, data):
        logging.info('initialized Secret obj')
        self.data = data
        self.no_states = 10101010
        self.no_actions = 125
        self.Q = np.zeros((self.no_states, self.no_actions))
        self.policy = np.zeros(self.no_states)

    def Q_learning(self):
        logging.info('-> Q-learning')
        for row in tqdm(self.data.iterrows(), total=len(self.data)):
            s, a, r, sp = row[1]
            self.Q[s - 1, a - 1] = self.Q[s - 1, a - 1] + 0.1*(r + 0.95 * self.Q[sp - 1, a - 1] - self.Q[s - 1, a - 1])

        self._update_policy()

    def _update_policy(self):
        logging.info('updating policy')
        for s in tqdm(range(self.no_states)):
            if not np.any(self.Q[s]):
                self.policy[s] = np.random.randint(1, 126)
            else:
                self.policy[s] = np.argmax(self.Q[s]) + 1

    def output_policy(self):
        logging.info('writing policy to file')
        with open('large.policy', 'w+') as f:
            f.writelines([str(int(x)) + '\n' for x in self.policy])


