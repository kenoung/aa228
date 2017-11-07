import logging
import random
import sys

import itertools
import tempfile

import graphviz
import networkx as nx
import os
import pandas as pd
import time
from networkx.drawing.nx_pydot import write_dot
from scipy.special import gammaln
import pylab as plt


def write_gph(G, filename):
    edges = []
    for key, values in nx.to_dict_of_lists(G).items():
        for value in values:
            edges.append('{},{}\n'.format(key,value))

    with open(filename, 'w') as f:
        f.writelines(edges)


def init_graph(data):
    G = nx.DiGraph()
    G.add_nodes_from(list(data.columns))
    return G


def draw_graph(G, filename):
    with tempfile.NamedTemporaryFile() as f:
        write_dot(G, f.name)
        img_file = graphviz.render('dot', 'png', f.name)
        os.rename(img_file, filename)

def compute(infile, outfile):
    data = pd.read_csv(infile)
    G = init_graph(data)

    start = time.time()
    while True:
        score = bayesian_score(G, data)
        k_two(G, data, 1)
        edge_direction_optimization(G, data)
        if score == bayesian_score(G, data):
            break

    write_gph(G, outfile)
    draw_graph(G, infile.replace('csv','png'))

    end = time.time()
    logging.info('start: {} | end: {} | time spent: {}'.format(start, end, end-start))


def local_bayesian_score(G, node, data):
    """
    Returns the local bayesian score of a node of a graph G given the data D
    """
    score = 0
    ri = max(data[node])
    parents = sorted(list(G.pred[node]))
    possible_parent_instantiations = itertools.product(*[data[p].unique() for p in parents])
    for inst in possible_parent_instantiations:
        temp = data.copy()
        for l in range(len(parents)):
            temp = temp[temp[parents[l]] == inst[l]]

        score += gammaln(ri) - gammaln(ri + len(temp))

        for k in range(ri):
            m = len(temp[temp[node] == k + 1])
            score += gammaln(1 + m)

    return score


def bayesian_score(G, data):
    return sum(local_bayesian_score(G, node, data) for node in G)


def k_two(G, data, parent_count):
    logging.info('running k2 algo')
    scores = [local_bayesian_score(G, node, data) for node in G]

    for idx, node in enumerate(G):
        logging.info('optimizing parent set for node [{}]'.format(node))
        parents = []

        while True:
            curr_score = sum(scores)
            curr_best_local_score = scores[idx]
            curr_best_parent = None

            for new_parent in G:
                Gtemp = G.copy()

                if new_parent == node or new_parent in Gtemp.pred[node] or node in Gtemp.pred[new_parent]:
                    continue

                Gtemp.add_edge(new_parent, node)
                new_local_score = local_bayesian_score(Gtemp, node, data)
                if nx.is_directed_acyclic_graph(Gtemp) and new_local_score > curr_best_local_score:
                    curr_best_local_score = new_local_score
                    curr_best_parent = new_parent

            if curr_best_parent:
                G.add_edge(curr_best_parent, node)
                parents.append(curr_best_parent)
                scores[idx] = curr_best_local_score
                logging.info('{} -> {}'.format(curr_score, sum(scores)))

                if len(parents) >= parent_count:
                    break

            else:
                if parents:
                    logging.info('added to {}: {}'.format(node, ' '.join(parents)))
                break


def edge_direction_optimization(G, data):
    logging.info('running edge direction optimization')
    scores = {node: local_bayesian_score(G, node, data) for node in G}

    for edge in list(G.edges):
        curr_score = sum(scores.values())
        Gtemp = G.copy()

        parent, child = edge
        parent_score = scores[parent]
        child_score = scores[child]
        Gtemp.remove_edge(*edge)
        Gtemp.add_edge(*edge[::-1])

        if not nx.is_directed_acyclic_graph(Gtemp):
            continue

        new_parent_score = local_bayesian_score(Gtemp, parent, data)
        new_child_score = local_bayesian_score(Gtemp, child, data)
        if new_parent_score + new_child_score > parent_score + child_score:
            logging.info('swap {} -> {}'.format(parent, child))
            G.remove_edge(*edge)
            G.add_edge(*edge[::-1])
            scores[parent] = new_parent_score
            scores[child] = new_child_score
            logging.info("{} -> {}".format(curr_score, sum(scores.values())))



def logging_config():
    numeric_level = getattr(logging, 'INFO')
    logging.basicConfig(level=numeric_level,
                        format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    logging_config()
    main()
