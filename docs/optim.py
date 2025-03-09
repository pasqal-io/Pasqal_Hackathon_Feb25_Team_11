import numpy as np
import matplotlib.pyplot as plt
import pulser
from pulser.waveforms import InterpolatedWaveform
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def evaluate_mapping(new_coords: np.ndarray, Q: np.ndarray, device: pulser.devices.Device):
    """
    Computes the deviation between the interaction matrix from given atom coordinates
    and the target QUBO matrix Q.

    Parameters:
    new_coords : np.ndarray  - Flattened 2D atom coordinates.
    Q : np.ndarray  - Target QUBO matrix.
    device : pulser.devices.Device  - Quantum device with interaction coefficient.

    Returns:
    float  - Frobenius norm of the matrix difference.
    """
    new_coords = np.reshape(new_coords, (len(Q), 2))

    # computing the matrix of the distances between all coordinate pairs
    new_Q = squareform(device.interaction_coeff / pdist(new_coords) ** 6)
    return np.linalg.norm(new_Q - Q)


def coords_optim(Q, verbose=False, device=pulser.DigitalAnalogDevice):
    """
    Optimizes atom coordinates to minimize deviation from the target QUBO matrix Q.

    Parameters:
    Q : np.ndarray  - Target QUBO matrix.
    verbose : bool  - If True, prints optimization details.
    device : pulser.DigitalAnalogDevice  - Quantum device for interaction coefficient computation.

    Returns:
    np.ndarray  - Optimized 2D atom coordinates.
    """
    best_res = None
    best_value = float("inf")

    for i in range(40):  # Number of different initializations
        np.random.seed(i)  # Change the seed for different initializations
        x0 = np.random.uniform(-30, 30, len(Q) * 2)
        res = minimize(
            evaluate_mapping,
            x0,
            args=(~np.eye(Q.shape[0], dtype=bool) * Q, device),
            method="L-BFGS-B",
            tol=1e-16,
            options={"maxiter": 200000},
        )

        if res.fun < best_value:
            best_res = res
            best_value = res.fun
    if verbose:
        print(best_res)
    coords = np.reshape(best_res.x, (len(Q), 2))
    return coords


def normalize_det_map(arr):
    """
    Creates DMM in the range of [0, 1] from the diagonal of the QUBO matrix.

    Parameters:
    arr : np.ndarray - Input array, contains diagonal of the QUBO matrix.

    Returns:
    np.ndarray - Normalized DMM map, or zeros if all values are equal.
    """
    arr = np.array(arr, dtype=float)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.zeros_like(
            arr
        )  # Avoid division by zero, return zeros if all values are equal (same as no DMM)
    return (arr - min_val) / (max_val - min_val)


def compute_min_interaction(Q, device):
    """
    Computes the minimum interaction distance based on the off-diagonal elements of Q.

    Parameters:
    Q : numpy.ndarray -  The QUBO matrix.
    device: pulser.DigitalAnalogDevice  - Quantum device for interaction coefficient computation.

    Returns:
    float or None - The minimum interaction distance, or None if no nonzero interactions exist.
    """
    off_diagonal_values = [
        abs(Q[i, j]) for i in range(Q.shape[0]) for j in range(Q.shape[1]) if i != j and Q[i, j] != 0
    ]

    if off_diagonal_values:  # Ensure there's at least one nonzero off-diagonal element
        Q_min = min(off_diagonal_values)
        R_min = (device.interaction_coeff / Q_min) ** (1 / 6)
        return R_min * 10
    else:
        print("No nonzero off-diagonal elements found.")
        return None


def get_cost_bitstring(bitstring, Q):
    """
    Computes the cost of a given bitstring based on the QUBO matrix Q.

    Parameters:
    bitstring : str  - Binary string representing a solution.
    Q : np.ndarray  - QUBO matrix.

    Returns:
    int - Computed cost value.
    """
    x = np.array(list(bitstring), dtype=int)
    cost = x.T @ Q @ x
    return cost


def create_interp_pulse(amp_params, det_params, delta_bounds, T):
    """
    Creates an interpolated pulse with specified amplitude and detuning parameters.

    Parameters:
    amp_params : list  - Amplitude interpolation points.
    det_params : list  - Detuning interpolation points.
    delta_bounds : tuple  - Boundary values for detuning.
    T : int  - Pulse duration in ns.

    Returns:
    pulser.Pulse - Generated pulse.
    """
    return pulser.Pulse(
        InterpolatedWaveform(T, [1e-9, *amp_params, 1e-9]),
        InterpolatedWaveform(T, [delta_bounds[0], *det_params, delta_bounds[1]]),
        0,
    )


def find_best(count_dict, Q):
    """
    Finds the bitstring with the lowest cost based on the QUBO matrix Q.

    Parameters:
    count_dict : dict  - Dictionary of measured bitstrings.
    Q : np.ndarray  - QUBO matrix.

    Returns:
    list - Best found bitstring (as an np.ndarray) and its corresponding cost (as a float).
    """
    res = list(count_dict.keys())
    b_init = np.array(list("0" * Q.shape[0]), dtype=int)
    cost_init = b_init.T @ Q @ b_init
    b_cost = [b_init, cost_init]
    for i in res:
        b = np.array(list(i), dtype=int)
        curr = b.T @ Q @ b
        if curr < b_cost[1]:
            b_cost[0] = b
            b_cost[1] = curr
    return b_cost


def plot_distribution(C, correct):
    """
    Plots the distribution of measured bitstrings, highlighting the correct ones.

    Parameters:
    C : dict  - Dictionary of bitstrings and their counts.
    correct : list  - List of correct bitstrings to highlight.

    Returns:
    None
    """
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    indexes = correct
    color_dict = {key: "r" if key in indexes else "g" for key in C}
    plt.figure(figsize=(12, 6))
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
    plt.xticks(rotation="vertical")
    plt.show()
    print("The correct bitstring(s):")
    print(correct)
