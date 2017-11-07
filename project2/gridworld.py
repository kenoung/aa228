import numpy as np
import matplotlib.pyplot as plt


def move_left(x, y):
    return x - 1, y


def move_right(x, y):
    return x + 1, y


def move_up(x, y):
    return x, y + 1


def move_down(x, y):
    return x, y - 1


ACTION_MAP = {
    move_left: 1,
    move_right: 2,
    move_up: 3,
    move_down: 4
}


class GridWorld(object):
    def __init__(self, N, R, discount):
        """
        Parameters
        ----------
        N : int
        side-length of the grid. i.e. grid size is N*N

        R: 2D N*N numpy array
        rewards for each given state

        discount : int
        discount factor
        """
        self.N = N
        self.R = R
        self.U = np.zeros((N, N))
        self.arrows = np.ones((N, N), dtype=np.int)
        self.iterations = 0
        self.discount = discount

    def T(self, x, y, a):
        pos = x + 1 + y * self.N
        valid_actions = self.get_valid_actions(pos)

        if a not in valid_actions:
            return []

        new_states = []
        for valid_action in valid_actions:
            if valid_action == a:
                new_states.append((0.6, *valid_action(x, y)))
            else:
                new_states.append((0.4 / (len(valid_actions) - 1), *valid_action(x, y)))
        return new_states

    def get_valid_actions(self, pos):
        actions = []
        if pos > self.N:
            actions.append(move_down)
        if pos % self.N != 1:
            actions.append(move_left)
        if pos % self.N != 0:
            actions.append(move_right)
        if pos < (self.N - 1) * self.N:
            actions.append(move_up)
        return actions

    def update(self, n=1):
        for _ in range(n):
            self.iterations += 1

            for x in range(self.N):
                for y in range(self.N):
                    self.U[x][y] = self.R[x][y] + 0.95 * max(
                        [sum([p * self.U[x2][y2] for (p, x2, y2) in self.T(x, y, a)])
                         for a in [move_left, move_right, move_up, move_down]])

        self._update_policy()

    def _update_policy(self):
        for x in range(self.N):
            for y in range(self.N):
                scores = [sum([p * self.U[x2][y2] for (p, x2, y2) in self.T(x, y, a)])
                          for a in [move_left, move_right, move_up, move_down]]
                self.arrows[x][y] = scores.index(max(scores)) + 1

    def show(self):
        # initialize fig
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)

        # position labels
        for i in range(self.N):
            for j in range(self.N):
                # label
                x = i + 0.5
                y = j + 0.5
                ax.text(x - 0.05 * len(str(int(self.U[i][j]))), y + 0.3, int(self.U[i][j]))

                # draw arrows
                xp, yp = [(-0.25, 0), (0.25, 0), (0, 0.25), (0, -0.25)][int(self.arrows[i][j] - 1)]
                ax.arrow(x, y - 0.1, xp, yp, fc="k", ec="k", head_width=0.1, head_length=0.1)

        ax.pcolormesh(self.U.T, cmap=plt.cm.get_cmap('viridis'), vmin=-np.max(self.U))

        # axes
        ax.set_ylim([0, self.N])
        ax.set_xlim([0, self.N])
        ax.set_yticks(range(self.N))
        ax.set_xticks(range(self.N))
        ax.grid(color='k')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    def output_policy(self):
        with open('small.policy', 'w+') as f:
            f.writelines([str(int(x)) + '\n' for x in self.arrows.T.reshape((100,))])
