#!/usr/bin/env python3
from enum import IntEnum
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

MIN_DEGREE = 3
NUM_POINTS = 15
NUM_DIM = 2
NUM_TESTS = 5

class Algs(IntEnum):
    alg1 = 0    
    alg2 = 1

def alg1(points, graph, min_deg=MIN_DEGREE):
    # min_deg < len(points)
    if min_deg >= len(points):
        return

    # geometric mean
    gmean = np.mean(points, 0)
    
    # map to closest point
    center = np.argmin(np.sum((points - gmean) ** 2, 1))

    sum = 0

    # make link to center for non-center nodes
    graph[:, center] = 1
    graph[center, center] = 0

    # add lengths
    lengths = np.sqrt(np.sum((points - points[center]) ** 2, 1))
    sum += np.sum(lengths)

    # remove center
    point_set = set(range(len(points)))
    point_set.remove(center)

    # make links to nearest nodes
    for p in point_set:
        # get adjacent nodes
        #adj = np.flatnonzero(graph[:, p] == 1)
        adj = np.flatnonzero(graph[:, p] > 0)

        # get n nearest points
        n = min_deg - (len(adj) + 1)

        if n <= 0:
            continue

        squared_lengths = np.sum((points - points[p]) ** 2, 1)

        squared_lengths[center] = np.inf
        squared_lengths[adj] = np.inf
        squared_lengths[p] = np.inf

        nearest = np.argpartition(squared_lengths, n)

        # add edges
        graph[p][nearest[:n]] = 1

        # add lengths
        lengths = np.sqrt(squared_lengths[nearest[:n]])
        sum += np.sum(lengths)

    print(graph)
    print(sum)

    return sum

def alg2(points, min_deg=MIN_DEGREE, num_means=-1):
    # initialize means
    if num_means <= 0:
        num_means = len(points) // (min_deg + 1)
        means = np.random.random((num_means, NUM_DIM))
    else:
        means = np.random.random((num_means, NUM_DIM))

    point_means = np.empty(len(points))
    m_means(points, means, point_means)
    print(means)

def min_metric(means):
    pass

def m_means(points, means, point_means):
    # map points to closest means
    #min_metric(means)

    # for point in points:

    # use expectation as mean
    pass

def plot(tests, graphs, sums):
    for alg in Algs:
        for i in range(len(graphs[alg])):
            pos = {p: point for p, point in enumerate(tests[i])}
            ax = plt.axes()
            G = nx.from_numpy_matrix(graphs[alg][i])
            nx.draw(G, pos, ax, node_size=50)
            ax.set_title(f'Algorithm {alg + 1}, Test {i}, Sum: {sums[alg][i]}')
            ax.tick_params(left=True, bottom=True,
                           labelleft=True, labelbottom=True)
            plt.axis("on")
            plt.savefig(f'alg{alg + 1}_graph{i}.png')
            plt.clf()

def main():
    tests = np.random.random((NUM_TESTS, NUM_POINTS, NUM_DIM))
    graphs = np.zeros((len(Algs), NUM_TESTS, NUM_POINTS, NUM_POINTS), dtype=np.uint8)
    sums = np.zeros((len(Algs), NUM_TESTS))
    for i in range(len(tests)):
        sums[Algs.alg1][i] = alg1(tests[i], graphs[Algs.alg1][i])
        #alg2(tests[i])

    plot(tests, graphs, sums)

if __name__ == '__main__':
    main()
