from random_generator import generate_random_points, get_max_capacity_args
from misc import calculate_and_visualise





def main(k, n, Random):
    max_capacity_N = 400
    max_capacity_K = 20

    if not Random:
        """
        some args sanity
        """
        if k is None or n is None:
            raise ValueError("Must specify k and n!")

        try:
            k = int(k)
            n = int(n)

        except Exception:
            raise ValueError("Both k and n need to be integers!")

        if k <= 0 or n <= 0:
            raise ValueError("Both k and n need to be positive!")

        fixed_k = k
    else:
        if k is not None or n is not None:
            print("Random specified, ignoring k,n values")
        k = get_max_capacity_args()[0]
        n = get_max_capacity_args()[1]
        fixed_k = None

    blobs, blobs_arr, d = generate_random_points(k, n)

    print(f"Using max capacity N: {max_capacity_N}, max capacity K: {max_capacity_K}")
    print(f"Running with n={n}, k={k}")
    calculate_and_visualise(blobs, blobs_arr, n, k, d, fixed_k=fixed_k)

    
if __name__ == "__main__":
    main("3", "20", True)
