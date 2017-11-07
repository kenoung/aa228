import logging

import pandas as pd
import time

from project2.gridworld import GridWorld
from project2.movingCar import MovingCar
from project2.secret import Secret

DATA_DIR = 'data'
SMALL = DATA_DIR + '/small.csv'
MEDIUM = DATA_DIR + '/medium.csv'
LARGE = DATA_DIR + '/large.csv'

def small():
    small_data = pd.read_csv(SMALL)

    # rewards for a given tile
    sp_r = small_data[['s', 'r']].drop_duplicates().sort_values('s').r.values.reshape((10, 10))

    # grid is of size N x N
    N = 10

    g = GridWorld(N, sp_r, 0.95)

    start = time.time()
    g.update(50)
    logging.info('took {}s for 50 iterations'.format(time.time() - start))
    g.output_policy()

def medium():
    med_data = pd.read_csv(MEDIUM)
    med_data['vel'] = med_data.s // 500
    med_data['pos'] = med_data.s % 500 - 1
    med_data['vel_p'] = med_data.sp // 500
    med_data['pos_p'] = med_data.sp % 500 - 1
    med_data['d_pos'] = med_data.pos - med_data.pos_p
    med_data['d_vel'] = med_data.vel - med_data.vel_p
    mc = MovingCar(med_data)
    mc.value_iteration(10)
    mc.output_policy()


def large():
    large_data = pd.read_csv(LARGE)
    secret = Secret(large_data)
    start = time.time()
    secret.Q_learning()
    logging.info('took {}s'.format(time.time() - start))
    secret.output_policy()

def logging_config():
    numeric_level = getattr(logging, 'INFO')
    logging.basicConfig(level=numeric_level,
                        format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

if __name__ == '__main__':
    logging_config()
    medium()