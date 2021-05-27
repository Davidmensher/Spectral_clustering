import argparse
import pandas as pd
import numpy as np

def arg_parser():
    parser = argparse.ArgumentParser(description="A program to compute kmeans algorithm")
    parser.add_argument("K", help="Number of clusters required", type=int)
    parser.add_argument("N", help="Number of observations in the file", type=int)
    parser.add_argument("d", help="Dimension of each observation and initial centroids", type=int)
    parser.add_argument("MAX_ITER", help="Maximum number of iterations of the K-means algorithm", type=int)
    parser.add_argument("filename", help="path for the observations file", type=str)

    return parser

def k_means_pp(K, N, d, observations):
    """
    this method chooses the K centroids according to the algorithm and returns them as ndarray.
    it also prints the indexes of the centroids by the order they have been chosen.
    """
    # Seed randomness
    np.random.seed(0)

    # randomly choose the first centroid
    last = np.random.choice(N, 1)

    centroids = np.array(observations[last])
    centroids_indexes = np.array([last])

    # initialize min distances
    min_distances = np.power((observations - last), 2).sum(axis=1)

    for j in range(1, K):
        # pool last centroid
        last = centroids[-1]

        # recompute min distances
        new_distances = np.power((observations - last), 2).sum(axis=1)
        min_distances = np.minimum(new_distances, min_distances)

        # sum of distances
        sum_of_wights = sum(min_distances)

        # computes the probability
        prob = [min_distances[i]/sum_of_wights for i in range(N)]

        # choose the centroid index using the probabilities calculated at prob
        rand = np.random.choice(N, 1, p=prob)
        added_cent = np.array(observations[rand])

        # add the chosen centroid to the centroids set
        centroids = np.concatenate((centroids, added_cent))

        # add the index to the index array
        centroids_indexes = np.append(centroids_indexes, rand)

    return centroids
