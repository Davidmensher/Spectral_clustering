"""
File for all output related code
"""
from kmeans_pp import k_means_pp
from kmeans import kmeans
import numpy as np
from NS_Clustering import calculate_ns_clustering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_input_file(blobs, blobs_arr):
    with open("data.txt", 'w') as fout:
        for clusters, ind in zip(blobs, blobs_arr):
            clusters = [str(point) for point in clusters]
            f_line = f"{','.join(clusters)},{ind}\n"

            fout.write(f_line)


def calculate_kmeans(blobs, k, n, d):
    # calculate kmeans with kmeanspp
    centroids = k_means_pp(k, n, d, np.array(blobs))
    d, c = kmeans(blobs, centroids.tolist(), k, n, d, 300)

    return d


def parse_kmeans(clusters, blobs, k, n, d):
    # helper function to parse returned algorithm values
    parsed_output = []
    for cluster in clusters:
        parse_row = []
        for point in clusters[cluster]:

            row = np.all(blobs == point, axis=1)
            for i, r in enumerate(row):
                if r:
                    parse_row.append(i)
        parsed_output.append(parse_row)

    return parsed_output


def generate_flatten(cluster_list, num_blobs):
    """
    Generate a helper array for jaccard
    """
    flatten_cluster = [0 for i in range(num_blobs)]

    for i, cluster in enumerate(cluster_list):
        for ind in cluster:
            flatten_cluster[ind] = i

    return flatten_cluster


def jaccard_distance(cluster_list, blobs_list):
    # Calculate jaccard distance
    num = 0
    denum = 0

    for i in range(len(blobs_list)):
        for j in range(i + 1, len(blobs_list)):
            if cluster_list[i] == cluster_list[j] and blobs_list[i] == blobs_list[j]:
                num += 1

            if cluster_list[i] == cluster_list[j] or blobs_list[i] == blobs_list[j]:
                denum += 1

    return num / denum


def parsed_string(parsed_output):
    """
    Helper function for clusters.txt
    """
    st = []
    for l in parsed_output:
        comma_parsed = ','.join([str(i) for i in l])
        st.append(comma_parsed)

    return '\n'.join(st)


def generate_output_file(blobs, parsed_output, parsed_ns_output, k, ns_k, n, d):
    with open("clusters.txt", 'w') as fout:
        fout.write(f"{str(k)}\n")

        parsed_ns_output = parsed_string(parsed_ns_output)
        fout.write(parsed_ns_output)

        fout.write('\n')

        parsed_output = parsed_string(parsed_output)
        fout.write(parsed_output)


def generate_3d_pdf(blobs, k_means_array, ns_array, k_jaccard, ns_jaccard, n, d, k, ns_k):
    with PdfPages("clusters.pdf") as pf:
        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(222, projection='3d')
        ax.set_title("K-means")

        bx = fig.add_subplot(221, projection='3d')
        bx.set_title("Normalized Spectral Clustering")

        colors = [np.random.rand(3, ) for i in range(k)]

        for i in range(len(k_means_array)):
            ax.scatter(blobs[i, 0], blobs[i, 1], blobs[i, 2], color=colors[k_means_array[i]], alpha=0.8)

        for i in range(len(ns_array)):
            bx.scatter(blobs[i, 0], blobs[i, 1], blobs[i, 2], color=colors[ns_array[i]], alpha=0.8)

        fig.text(.5, 0.2, "Data was generated from the values:\n"
                          f"n = {len(blobs)} , k = {k}\n"
                          f"The k that was used for both algorithms was {ns_k}\n"
                          f"The Jaccard measure for Spectral Clustering: {ns_jaccard:.2f}\n"
                          f"The Jaccard measure for K-means: {k_jaccard:.2f}", ha='center', fontsize="14")
        pf.savefig(fig, bbox_inches='tight')


def generate_2d_pdf(blobs, k_means_array, ns_array, k_jaccard, ns_jaccard, n, d, k, ns_k):
    with PdfPages("clusters.pdf") as pf:
        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(222)
        ax.set_title("K-means")

        bx = fig.add_subplot(221)
        bx.set_title("Normalized Spectral Clustering")

        colors = [np.random.rand(3, ) for i in range(ns_k)]

        for i in range(len(k_means_array)):
            ax.scatter(blobs[i, 0], blobs[i, 1], color=colors[k_means_array[i]], alpha=0.8)

        for i in range(len(ns_array)):
            bx.scatter(blobs[i, 0], blobs[i, 1], color=colors[ns_array[i]], alpha=0.8)

        fig.text(.5, 0.2, "Data was generated from the values:\n"
                          f"n = {len(blobs)} , k = {k}\n"
                          f"The k that was used for both algorithms was {ns_k}\n"
                          f"The Jaccard measure for Spectral Clustering: {ns_jaccard:.2f}\n"
                          f"The Jaccard measure for K-means: {k_jaccard:.2f}", ha='center', fontsize="14")
        pf.savefig(fig, bbox_inches='tight')


def calculate_and_visualise(blobs, blobs_arr, n, k, d, fixed_k=None):
    generate_input_file(blobs, blobs_arr)

    ns_k, parsed_ns_output = calculate_ns_clustering(blobs, n, fixed_k=fixed_k)
    kmeans_cluster = calculate_kmeans(blobs.tolist(), ns_k, n, d)

    parsed_kmeans_output = parse_kmeans(kmeans_cluster, blobs, ns_k, n, d)
    generate_output_file(blobs, parsed_kmeans_output, parsed_ns_output, ns_k, ns_k, n, d)

    k_flatten = generate_flatten(parsed_kmeans_output, n)
    ns_flatten = generate_flatten(parsed_ns_output, n)

    k_jaccard = jaccard_distance(k_flatten, blobs_arr)
    ns_jaccard = jaccard_distance(ns_flatten, blobs_arr)

    if d == 3:
        generate_3d_pdf(blobs, k_flatten, ns_flatten, k_jaccard, ns_jaccard, n, d, k, ns_k)

    else:
        generate_2d_pdf(blobs, k_flatten, ns_flatten, k_jaccard, ns_jaccard, n, d, k, ns_k)
