import os

import dimod
import numpy as np
from patchify import patchify
from pyqubo import Binary
from scipy import ndimage

from qubo.utils import plot_patches, plot_solution, plot_tiff
from utils import get_project_root, read_tiff_file


def pyqubo_solver(
    qubo_matrix,
    num_atoms,
    exact_solver=False,
    verbose=False,
):
    n = num_atoms
    cost = 0
    decision_vars = {}
    for i in range(n):
        var_name = f"x{i}"
        decision_vars[var_name] = Binary(var_name)

    for i in range(n):
        for j in range(n):
            cost += qubo_matrix[i, j] * decision_vars[f"x{i}"] * decision_vars[f"x{j}"]

    model = cost.compile()
    qubo, offset = model.to_qubo()

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

    if exact_solver:
        sampler = dimod.ExactSolver()
    else:
        sampler = dimod.SimulatedAnnealingSampler()

    response = sampler.sample(bqm)
    best_sample = response.first.sample
    if verbose:
        print("QUBO solver output:")
        print("Best sample:", best_sample)
        print("QUBO offset:", offset)

    return best_sample


def compute_neighbours(padded_features, row, col):
    neighbours = np.array(
        [
            padded_features[row - 1, col],  # Top
            padded_features[row + 1, col],  # Bottom
            padded_features[row, col - 1],  # Left
            padded_features[row, col + 1],  # Right
        ]
    )
    neighbours = np.nan_to_num(neighbours, nan=-1e-2)
    return neighbours.sum() / len(neighbours)


def compute_qubo_matrix(
    patch_features,
    patches_per_dim,
    resource_limit=2,
    resource_penalty=1e0,
    image_name="Q",
    results_path="results",
    save_output=False,
):
    n = patches_per_dim * patches_per_dim

    padded_features = np.pad(patch_features, pad_width=1)

    norm_distance = (patches_per_dim**2 + patches_per_dim**2) ** 0.5

    # TODO The numbers should be -20 < diagonal < 0 and 0 < off diagonal < 20
    qubo_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            row, col = i // patches_per_dim + 1, i % patches_per_dim + 1
            neighbour_row, neighbour_col = (
                j // patches_per_dim + 1,
                j % patches_per_dim + 1,
            )

            q_val = 0.0
            if i == j:
                if np.isnan(padded_features[row, col]):
                    q_val = -10e-9
                else:
                    q_val = padded_features[row, col]
                    q_val += compute_neighbours(padded_features, row, col)
                    if q_val == 0:
                        q_val = -1e-9
                    else:
                        q_val += resource_penalty * (1 - 2 * resource_limit)
            elif i < j:
                q_val = resource_penalty * 2
                distance = ((row - neighbour_row) ** 2 + (col - neighbour_col) ** 2) ** 0.5
                q_val -= distance / norm_distance

            qubo_matrix[i, j] = q_val

    qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2

    if save_output:
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, image_name), "wb") as f:
            np.save(f, qubo_matrix)

    return qubo_matrix


def extract_wildfire_supression_features(
    patches: np.ndarray,
    patches_per_dim: int,
    features_to_idx: dict,
    fire_threshold: float = 0.8,
):
    # Tranpose to get patches_per_dim, patches_per_dim, channels, height, width
    patches = patches.transpose(1, 2, 0, 3, 4)

    patch_scores = np.zeros((patches_per_dim, patches_per_dim))

    idx_fire = features_to_idx["active fire"]
    idx_wind_speed = features_to_idx["wind speed"]
    idx_precip = features_to_idx["total precipitation"]
    idx_pdsi = features_to_idx["pdsi"]  # Palmer Drought Severity Index
    idx_erc = features_to_idx["energy release component"]
    # idx_slope = features_to_idx["slope"]

    # For vegetation dryness we might also consider NDVI/EVI.
    idx_ndvi = features_to_idx["NDVI_last"]

    for row in range(patches_per_dim):
        for col in range(patches_per_dim):
            patch = patches[row, col]

            fire_band = patch[idx_fire]
            total_pixels = fire_band.size

            mean_wind_speed = patch[idx_wind_speed].mean()
            mean_precip = patch[idx_precip].mean()
            mean_pdsi = patch[idx_pdsi].mean()
            mean_erc = patch[idx_erc].mean()
            # mean_slope = patch[idx_slope].mean()
            mean_ndvi = patch[idx_ndvi].mean()

            # Construct synergy terms or intermediate "risk" factors:
            #     Example synergy: dryness × wind × slope
            #     - Lower pdsi => drier conditions => more fire spread
            #     - Higher ERC => more potential for intense fire
            #     - NDVI might be negative or small for dryness (some areas invert NDVI).
            #       But let's treat *lower NDVI => drier vegetation => higher risk*
            #       So we can do dryness_factor = - mean_pdsi + mean_erc - mean_ndvi
            dryness_factor = (
                mean_pdsi + mean_precip - mean_erc - mean_ndvi  # negative if PDSI is large => less dryness
            )

            spread_potential = dryness_factor - mean_wind_speed  # - slope

            fraction_fire = np.count_nonzero(fire_band) / total_pixels

            if fraction_fire > fire_threshold:
                patch_score = np.nan
            else:
                patch_score = fraction_fire * spread_potential

            patch_scores[row, col] = patch_score

    return patch_scores


def image_segment(image, filled_image):
    eroded = ndimage.binary_erosion(image, structure=np.ones((2, 2)))
    boundary = filled_image - eroded  # Boundaries are the difference

    labeled, num_features = ndimage.label(image)

    segmented = np.zeros_like(image, dtype=int)

    segmented[boundary == 1] = 1

    for label in range(1, num_features + 1):
        mask = labeled == label  # Get the connected component
        if np.any(boundary[mask]):  # Ensure it's a closed region
            segmented[mask] = -1
        segmented[boundary == 1] = 1  # Ensure boundaries remain 1

    return segmented


def extract_fire_region(image, expand_border_pixels=10):
    n_channels, height, width = image.shape

    fire = image[-1]

    rows = np.any(fire, axis=1)
    cols = np.any(fire, axis=0)

    # Extract the smallest bounding box containing all 1s
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    row_min = max(row_min - expand_border_pixels, 0)
    row_max = min(row_max + expand_border_pixels, height - 1)
    col_min = max(col_min - expand_border_pixels, 0)
    col_max = min(col_max + expand_border_pixels, width - 1)

    # Extract the smallest bounding box containing all 1s
    cropped_image = image[:, row_min : row_max + 1, col_min : col_max + 1]

    return cropped_image


def preprocess_image(image, features_to_idx, show_image=False):
    image = np.nan_to_num(image, nan=0)  # maybe smth else should be done

    channels = image[:-1]
    max_elems = np.absolute(channels).max(axis=(-2, -1)) + 1e-9
    channels /= max_elems[:, np.newaxis, np.newaxis]
    image[:-1] = channels

    fire_mask = image[-1]
    fire_mask[fire_mask > 0] = 1

    # Perform morphological operation to remove noise and close fires
    closed_fire = ndimage.binary_closing(fire_mask, structure=np.ones((30, 30))).astype(np.uint8)
    dilated_fire = ndimage.binary_dilation(closed_fire, structure=np.ones((5, 5))).astype(np.int8)

    # Segment fire border and fire interior
    image[-1] = image_segment(closed_fire, dilated_fire)

    cropped_image = extract_fire_region(image)

    if show_image:
        plot_tiff(cropped_image, [label for label in features_to_idx])

    return cropped_image


def extract_patches(image, features_to_idx, num_patches=16, show_patches=False):
    num_channels, height, width = image.shape

    patches_per_dim = int(num_patches**0.5)
    patch_size = (
        height // patches_per_dim,
        width // patches_per_dim,
    )
    patches = np.zeros(
        (
            num_channels,
            patches_per_dim,
            patches_per_dim,
            patch_size[0],
            patch_size[1],
        )
    )
    for chan_idx in range(num_channels):
        patches[chan_idx] = patchify(image[chan_idx], patch_size, step=patch_size)

    if show_patches:
        plot_patches(patches[-1], patches_per_dim)

    patch_features = extract_wildfire_supression_features(patches, patches_per_dim, features_to_idx)

    return patch_features


def combine_patches(patch_scores_horizon, patches_per_dim, discount_factor=0.7):
    combined_patch_scores = np.zeros((patches_per_dim, patches_per_dim))

    for patch_scores in reversed(patch_scores_horizon):
        combined_patch_scores = combined_patch_scores * discount_factor + patch_scores

    return combined_patch_scores


def _example_usage():
    sample_folder_path = "fire_25547912"

    data_path = os.path.join(get_project_root(), "data", "sample_source_dataset", sample_folder_path)
    results_path = os.path.join(get_project_root(), "results", "qubo_example")

    num_patches = 16
    patches_per_dim = int(num_patches**0.5)
    exact_solver = num_patches <= 16

    last_image_idx = 9
    horizon_length = 2

    image_paths = [
        f"2021-09-0{idx}.tif" for idx in range(last_image_idx - horizon_length + 1, last_image_idx + 1)
    ]

    patches_horizon = []
    for image_path in image_paths:
        image_path = os.path.join(data_path, image_path)
        _, image, feature_labels = read_tiff_file(image_path)

        features_to_idx = {label: idx for idx, label in enumerate(feature_labels)}

        image = preprocess_image(image, features_to_idx, show_image=False)
        patches = extract_patches(image, features_to_idx, num_patches=num_patches, show_patches=True)

        patches_horizon.append(patches)

    patch_scores = combine_patches(patches_horizon, patches_per_dim)

    qubo_matrix = compute_qubo_matrix(
        patch_scores,
        patches_per_dim,
        image_name=image_paths[0].split(".")[0],
        results_path=results_path,
    )

    solution = pyqubo_solver(
        qubo_matrix,
        num_atoms=num_patches,
        exact_solver=exact_solver,
    )

    plot_solution(solution, patches_per_dim)


if __name__ == "__main__":
    _example_usage()
