from sklearn.datasets import make_blobs
import random

def get_max_capacity_args():
    max_capacity_N = 400
    max_capacity_K = 20

    n_bound = max_capacity_N // 2
    k_bound = max_capacity_K // 2

    return random.randint(k_bound, max_capacity_K), random.randint(n_bound, max_capacity_N)

def generate_random_points(k, n):
    n_features = random.randint(2, 3)
    blobs = make_blobs(n_samples=n, n_features=n_features, centers=k)

    return blobs[0], blobs[1], n_features
