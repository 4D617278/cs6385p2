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

    # make link to center for non-center nodes
    graph[:, center] = 1
    graph[center, center] = 0

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

        distances = np.sum((points - points[p]) ** 2, 1)

        distances[center] = np.inf
        distances[adj] = np.inf
        distances[p] = np.inf

        nearest = np.argpartition(distances, n)

        # add edges
        graph[p][nearest[:n]] = 1

    print(graph)

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

def plot(tests, graphs):
    for alg in Algs:
        for i in range(len(graphs[alg])):
            pos = {p: point for p, point in enumerate(tests[i])}
            ax = plt.axes()
            G = nx.from_numpy_matrix(graphs[alg][i])
            nx.draw(G, pos, ax, node_size=50)
            ax.set_title(f'Algorithm {alg}, Test {i}')
            ax.tick_params(left=True, bottom=True,
                           labelleft=True, labelbottom=True)
            plt.axis("on")
            plt.savefig(f'alg{alg + 1}_graph{i}.png')
            plt.clf()

def main():
    tests = np.random.random((NUM_TESTS, NUM_POINTS, NUM_DIM))
    graphs = np.zeros((len(Algs), NUM_TESTS, NUM_POINTS, NUM_POINTS), dtype=np.uint8)
    for i in range(len(tests)):
        alg1(tests[i], graphs[Algs.alg1][i])
        #alg2(tests[i])

    plot(tests, graphs)

if __name__ == '__main__':
    main()
