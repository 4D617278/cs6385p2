#!/usr/bin/env python3
from enum import IntEnum
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

MAX_DIAM = 4
MIN_DEGREE = 3
NUM_POINTS = 150
NUM_DIM = 2
NUM_TESTS = 10

class Algs(IntEnum):
    alg1 = 0    
    alg2 = 1

def alg1(points, graph, min_deg=MIN_DEGREE, max_diam=MAX_DIAM):
    # min_deg < len(points)
    if min_deg >= len(points):
        return

    # max_diam >= 4 for (n - min_deg - 1) proof
    if max_diam < 4:
        return

    # compute lengths between each point
    lengths = np.sqrt(np.sum((points[:, None] - points[None, :]) ** 2, 2))

    # do not allow self-loops
    np.fill_diagonal(lengths, np.inf)

    # find point with min length to (n - min_deg - 1) nodes
    n = len(points) - min_deg - 1
    nearest = np.argpartition(lengths, n)[:, :n]
    indices = np.arange(len(points))[:, None]
    min_lengths = lengths[indices, nearest]
    min_length_sums = np.sum(min_lengths, 1)
    center = np.argmin(min_length_sums)

    # make link to center for (n - min_deg - 1) nearest nodes
    graph[nearest[center], center] = 1

    # add length sum
    sum = min_length_sums[center]

    # order by largest link length sum for m nearest nodes
    n = min_deg
    nearest = np.argpartition(lengths, n)[:, :n]
    min_lengths = lengths[indices, nearest]
    min_length_sums = np.sum(min_lengths, 1)
    sorted_points = np.argsort(min_length_sums)[::-1]

    # make links to nearest nodes
    for p in sorted_points:
        # get adjacent nodes
        edges = graph[:, p] > 0
        edges[center] = graph[p, center] > 0
        adj = np.flatnonzero(edges)

        # get n nearest points
        n = min_deg - len(adj)

        if n <= 0:
            continue

        lengths_copy = np.copy(lengths[p])
        lengths_copy[adj] = np.inf
        nearest = np.argpartition(lengths_copy, n)[:n]

        # add edges
        graph[p][nearest] = 1

        # add lengths
        sum += np.sum(lengths_copy[nearest])

    return sum

def alg2(points, graph, min_deg=MIN_DEGREE, max_diam=MAX_DIAM):
    # min_deg < len(points)
    if min_deg >= len(points):
        return

    # geometric center
    gcenter = np.mean(points, 0)

    # map to closest point
    center = np.argmin(np.sum((points - gcenter) ** 2, 1))

    # compute lengths between each point
    lengths = np.sqrt(np.sum((points[:, None] - points[None, :]) ** 2, 2))

    num_outer_layers = max_diam // 2

    # sort by decreasing distance
    sorted = np.argsort(lengths[center])[::-1]

    width = math.ceil((len(sorted) - 1) / num_outer_layers)

    sum = 0

    # do not allow self-loops
    np.fill_diagonal(lengths, np.inf)

    start = (len(sorted) - 1) - width * num_outer_layers

    for i in range(num_outer_layers):
        lower = max(0, start + i * width)
        upper = lower + width
        cur_layer = sorted[lower:upper]
        next_layer = sorted[upper:upper + width]

        # make links to next layer and nearest nodes
        for p in cur_layer:
            # link to closest node in next layer
            next = sorted[upper + np.argmin(lengths[p][next_layer])]
            graph[p][next] = 1
            sum += np.sum(lengths[p][next])

            # get adjacent nodes
            edges = graph[:, p] > 0
            edges[next] = 1
            adj = np.flatnonzero(edges)

            # get n nearest points
            n = min_deg - len(adj)

            if n <= 0:
                continue

            # link to nearest nodes
            lengths_copy = np.copy(lengths[p])
            lengths_copy[adj] = np.inf
            nearest = np.argpartition(lengths_copy, n)[:n]

            graph[p][nearest] = 1

            sum += np.sum(lengths_copy[nearest])

    # check number of links
    adj = np.flatnonzero(graph[:, center] > 0)
    n = min_deg - len(adj)

    if n > 0:
        # add edges from center
        lengths[adj] = np.inf
        nearest = np.argpartition(lengths[center], n)[:n]
        graph[center][nearest] = 1
        sum += np.sum(lengths[center][nearest])

    return sum

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
    graphs = np.zeros((len(Algs), NUM_TESTS, NUM_POINTS, NUM_POINTS), np.uint8)
    sums = np.zeros((len(Algs), NUM_TESTS))
    for i in range(len(tests)):
        sums[Algs.alg1][i] = alg1(tests[i], graphs[Algs.alg1][i])
        sums[Algs.alg2][i] = alg2(tests[i], graphs[Algs.alg2][i])

    print(f'Mean: {np.mean(sums[Algs.alg1])}')
    print(f'Mean: {np.mean(sums[Algs.alg2])}')
    plot(tests, graphs, sums)

if __name__ == '__main__':
    main()
