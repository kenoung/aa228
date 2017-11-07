import logging
import numpy as np
import time
from scipy.sparse import dok_matrix


class MovingCar(object):
    def __init__(self, data):
        self.data = data
        self.R = self._make_reward_matrix()
        self.T = self._make_transition_matrix()
        self.U = np.zeros(50000)
        self.Q = np.zeros(50000)
        self.update_progress = []

    def _make_reward_matrix(self):
        R_1, R_2, R_3, R_4, R_5, R_6, R_7 = [np.zeros(50000) for _ in range(7)]
        R_1.fill(-225)
        R_2.fill(-100)
        R_3.fill(-25)
        R_4.fill(0)
        R_5.fill(-25)
        R_6.fill(-100)
        R_7.fill(-225)
        for arr in [R_1, R_2, R_3, R_4, R_5, R_6, R_7]:
            arr[[True if idx % 500 + 0.3 * idx // 500 > 475 else False for idx in range(1, 50001)]] = 100000
        return [R_1, R_2, R_3, R_4, R_5, R_6, R_7]

    def _make_transition_matrix(self):
        transition_data = self.data.copy()
        transition_data = transition_data[transition_data.d_pos < 20] # re <move outliers
        transitions = []
        for a in range(1, 8):
            print('generating T_{}...'.format(a))
            subset = transition_data[transition_data.a == a].copy()
            change_in_velocity = subset.groupby(subset.pos // 10).d_vel.apply(lambda x: x.value_counts() / len(x))
            change_in_position = subset.groupby(subset.vel // 10).d_pos.apply(lambda x: x.value_counts() / len(x))
            transition_matrix = dok_matrix((50000, 50000))
            for s in range(50000):

                pos = s % 500
                vel = s // 500

                # absorbing state
                if pos + 0.3 * vel > 475:
                    continue

                # hit wall
                if pos + 0.35 * vel <=16:
                    transition_matrix[s, 25000] += 1
                    continue

                pos_idx = pos // 10
                pos_idx = max(min(change_in_velocity.index)[0], pos_idx)
                pos_idx = min(max(change_in_velocity.index)[0], pos_idx)
                while pos_idx not in change_in_velocity:
                    pos_idx += 1
                dv_table = change_in_velocity[pos_idx]

                vel_idx = vel // 10
                vel_idx = max(min(change_in_position.index)[0], vel_idx)
                vel_idx = min(max(change_in_position.index)[0], vel_idx)
                while vel_idx not in change_in_position:
                    vel_idx += 1
                dp_table = change_in_position[vel_idx]

                for dp_pair in zip(dp_table.index, dp_table):
                    for dv_pair in zip(dv_table.index, dv_table):
                        dp, proba_dp = dp_pair
                        dv, proba_dv = dv_pair
                        sp = max(0, min((pos + dp) + (vel + dv) * 500, 49999))
                        transition_matrix[s, sp] += proba_dp * proba_dv

            transitions.append(transition_matrix.tocsr())
        return transitions

    def value_iteration(self, ep=0.1):
        start = time.time()
        while True:
            Up = np.maximum.reduce([self.R[i] + self.T[i].dot(self.U) for i in range(7)])
            diff = (Up - self.U).max()
            self.update_progress.append(diff)
            self.U = Up
            if diff < ep:
                logging.info(
                    'Converged after {} iterations in {}s'.format(len(self.update_progress), time.time() - start))
                break

        self._update_policy()

    def _update_policy(self):
        self.Q = np.argmax(np.vstack([self.R[i] + self.T[i].dot(self.U) for i in range(7)]), axis=0) + 1

    def output_policy(self):
        logging.info('writing policy to file')
        with open('medium.policy', 'w+') as f:
            f.writelines([str(int(x)) + '\n' for x in self.Q])

