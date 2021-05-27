import argparse

def l2_norm(vec1, vec2):
	"""
	Computes ||vec1 - vec2|| ^ 2
	"""
	if len(vec1) != len(vec2):
		raise ValueError("Vectors are from different sizes!")

	norm = sum((v1 - v2) ** 2 for v1, v2 in zip(vec1, vec2))
	return norm

def get_min_centroid(vec, centorids):
	"""
	Gets the centroid that minimize l2_norm(vec, cent)
	"""

	# a little hack using lambda function, to counter the need of two args for l2_norm()
	return tuple(min(centorids, key=lambda cent: l2_norm(cent, vec)))

def initialize_helper_dict(centorids):
	"""
	Initialize a dictionary in the following way:
	entrance i will be as follows:
	cent_i: []
	"""
	help_dict = {}
	for cent in centorids:
		help_dict[tuple(cent)] = []

	return help_dict

def update_centroid(cluster, d):
	# sum cluster
	sum_vec = [0 for i in range(d)]
	for obs in cluster:
		sum_vec = [v1 + v2 for v1, v2 in zip(sum_vec, obs)]

	# divide vector by size
	cent = [(v / len(cluster)) for v in sum_vec]
	return tuple(cent)

def kmeans(observations, centroids_p, K, N, d, MAX_ITER):
	"""
	For some reason, our c implementation was slower :(
	"""
	centroids = centroids_p
	data_points = observations

	for i in range(MAX_ITER):
		help_dict = initialize_helper_dict(centroids)
		# cluster create loop
		for point in data_points:
			cent = get_min_centroid(point, centroids)
			help_dict[cent].append(point)

		# update centroids loop
		updated_centroid = [update_centroid(help_dict[tuple(cent)], d) for cent in centroids]
		
		# terminating condition
		if centroids == updated_centroid:
			break

		centroids = updated_centroid

	return help_dict, data_points
