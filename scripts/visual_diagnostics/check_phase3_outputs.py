#!/usr/bin/env python3
"""
Visual diagnostics for Phase 3 diffuse scattering outputs verification.

This script takes outputs from Phase 3 (GlobalVoxelGrid definition, refined scaling
parameters, and merged voxel data) and generates a series of diagnostic plots and
summary reports to visually verify the correctness of the voxelization, relative
scaling, and merging processes.

Key diagnostic outputs include:
1. Global voxel grid summary (text + conceptual visualization)
2. Voxel occupancy/redundancy analysis (heatmap slices, histogram)
3. Relative scaling model parameter plots (per-still scales, resolution smoother)
4. Merged voxel data visualization (intensity slices, radial averages, I/sigma)
5. Comprehensive summary report

Author: DiffusePipe
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib

matplotlib.use("Agg")  # Set early for non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Default maximum number of points to display in scatter plots for performance
DEFAULT_MAX_SCATTER_POINTS = 50000

# Import project utilities
try:
    from plot_utils import (
        plot_3d_grid_slice,
        plot_radial_average,
        plot_parameter_vs_index,
        plot_smoother_curve,
        ensure_output_dir,
        close_all_figures,
        setup_logging_for_plots,
    )
except ImportError as e:
    print(f"Error importing plot utilities: {e}")
    print("Please ensure you're running from the scripts/visual_diagnostics directory.")
    sys.exit(1)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visual diagnostics for Phase 3 diffuse scattering outputs verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python check_phase3_outputs.py \\
    --grid-definition-file phase3_outputs/global_voxel_grid_definition.json \\
    --scaling-model-params-file phase3_outputs/refined_scaling_model_params.json \\
    --voxel-data-file phase3_outputs/voxel_data_relative.npz \\
    --output-dir phase3_diagnostics

  # With optional additional data
  python check_phase3_outputs.py \\
    --grid-definition-file grid_def.json \\
    --scaling-model-params-file scaling_params.json \\
    --voxel-data-file voxel_data.npz \\
    --output-dir diagnostics \\
    --experiments-list-file experiments_list.txt \\
    --corrected-pixel-data-dir pixel_data_dirs.txt \\
    --max-plot-points 25000 \\
    --verbose
        """,
    )

    # Required arguments
    parser.add_argument(
        "--grid-definition-file",
        type=str,
        required=True,
        help="Path to GlobalVoxelGrid definition JSON file",
    )
    parser.add_argument(
        "--scaling-model-params-file",
        type=str,
        required=True,
        help="Path to refined scaling model parameters JSON file",
    )
    parser.add_argument(
        "--voxel-data-file",
        type=str,
        required=True,
        help="Path to VoxelData_relative NPZ/HDF5 file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for diagnostic plots and reports",
    )

    # Optional arguments
    parser.add_argument(
        "--experiments-list-file",
        type=str,
        help="Path to file containing list of experiment (.expt) file paths (optional)",
    )
    parser.add_argument(
        "--corrected-pixel-data-dir",
        type=str,
        help="Path to file containing list of corrected pixel data directories (optional)",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=DEFAULT_MAX_SCATTER_POINTS,
        help="Maximum number of points to display in scatter plots for performance",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def validate_input_files(args: argparse.Namespace) -> None:
    """Validate that all required input files exist."""
    required_files = [
        args.grid_definition_file,
        args.scaling_model_params_file,
        args.voxel_data_file,
    ]

    optional_files = [args.experiments_list_file, args.corrected_pixel_data_dir]

    # Check required files
    for file_path in required_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Check optional files
    for file_path in optional_files:
        if file_path and not Path(file_path).exists():
            logger.warning(f"Optional file not found: {file_path}")


def load_grid_definition(grid_file_path: str) -> Dict[str, Any]:
    """
    Load GlobalVoxelGrid definition from JSON file.

    Args:
        grid_file_path: Path to grid definition JSON file

    Returns:
        Dictionary containing grid definition parameters
    """
    try:
        with open(grid_file_path, "r") as f:
            grid_def = json.load(f)

        logger.info(f"Loaded grid definition from {grid_file_path}")

        # Validate required keys
        required_keys = [
            "crystal_avg_ref",
            "hkl_bounds",
            "ndiv_h",
            "ndiv_k",
            "ndiv_l",
            "total_voxels",
        ]
        for key in required_keys:
            if key not in grid_def:
                raise KeyError(f"Required key '{key}' not found in grid definition")

        return grid_def

    except Exception as e:
        logger.error(f"Failed to load grid definition from {grid_file_path}: {e}")
        raise


def load_scaling_parameters(scaling_file_path: str) -> Dict[str, Any]:
    """
    Load refined scaling model parameters from JSON file.

    Args:
        scaling_file_path: Path to scaling parameters JSON file

    Returns:
        Dictionary containing scaling parameters
    """
    try:
        with open(scaling_file_path, "r") as f:
            scaling_params = json.load(f)

        logger.info(f"Loaded scaling parameters from {scaling_file_path}")

        # Validate required keys
        required_keys = ["refined_parameters", "refinement_statistics"]
        for key in required_keys:
            if key not in scaling_params:
                raise KeyError(f"Required key '{key}' not found in scaling parameters")

        return scaling_params

    except Exception as e:
        logger.error(f"Failed to load scaling parameters from {scaling_file_path}: {e}")
        raise


def load_voxel_data(voxel_file_path: str) -> Dict[str, np.ndarray]:
    """
    Load VoxelData_relative from NPZ or HDF5 file.

    Args:
        voxel_file_path: Path to voxel data file

    Returns:
        Dictionary containing voxel data arrays
    """
    try:
        if voxel_file_path.endswith(".npz"):
            data = np.load(voxel_file_path)
            result = {key: data[key] for key in data.files}
        elif voxel_file_path.endswith(".hdf5") or voxel_file_path.endswith(".h5"):
            import h5py

            result = {}
            with h5py.File(voxel_file_path, "r") as f:
                for key in f.keys():
                    result[key] = f[key][:]
        else:
            raise ValueError("Voxel data file must be .npz or .hdf5/.h5 format")

        logger.info(f"Loaded voxel data from {voxel_file_path}")

        # Validate required keys
        required_keys = [
            "voxel_indices",
            "H_center",
            "K_center",
            "L_center",
            "q_center_x",
            "q_center_y",
            "q_center_z",
            "q_magnitude_center",
            "I_merged_relative",
            "Sigma_merged_relative",
            "num_observations",
        ]
        for key in required_keys:
            if key not in result:
                raise KeyError(f"Required key '{key}' not found in voxel data")

        logger.info(f"Voxel data contains {len(result['voxel_indices'])} voxels")
        return result

    except Exception as e:
        logger.error(f"Failed to load voxel data from {voxel_file_path}: {e}")
        raise


def generate_grid_summary(grid_def: Dict[str, Any], output_dir: Path) -> None:
    """
    Generate grid summary text file and conceptual visualization.

    Args:
        grid_def: Grid definition dictionary
        output_dir: Output directory for files
    """
    logger.info("Generating grid summary")

    # Text summary
    summary_text = []
    summary_text.append("=== Global Voxel Grid Summary ===\n")

    # Crystal parameters
    crystal_info = grid_def["crystal_avg_ref"]
    if "unit_cell_params" in crystal_info:
        uc_params = crystal_info["unit_cell_params"]
        summary_text.append(
            f"Average Unit Cell: a={uc_params[0]:.3f}, b={uc_params[1]:.3f}, c={uc_params[2]:.3f}"
        )
        summary_text.append(
            f"                   α={uc_params[3]:.2f}°, β={uc_params[4]:.2f}°, γ={uc_params[5]:.2f}°"
        )

    if "space_group" in crystal_info:
        summary_text.append(f"Space Group: {crystal_info['space_group']}")

    summary_text.append("")

    # HKL bounds
    hkl_bounds = grid_def["hkl_bounds"]
    summary_text.append("HKL Bounds:")
    summary_text.append(f"  H: {hkl_bounds['h_min']} to {hkl_bounds['h_max']}")
    summary_text.append(f"  K: {hkl_bounds['k_min']} to {hkl_bounds['k_max']}")
    summary_text.append(f"  L: {hkl_bounds['l_min']} to {hkl_bounds['l_max']}")
    summary_text.append("")

    # Voxel dimensions
    ndiv_h, ndiv_k, ndiv_l = grid_def["ndiv_h"], grid_def["ndiv_k"], grid_def["ndiv_l"]
    summary_text.append("Voxel Divisions:")
    summary_text.append(f"  ndiv_h: {ndiv_h}")
    summary_text.append(f"  ndiv_k: {ndiv_k}")
    summary_text.append(f"  ndiv_l: {ndiv_l}")
    summary_text.append("")

    # Total voxels
    total_voxels = grid_def["total_voxels"]
    summary_text.append(f"Total Voxels: {total_voxels:,}")

    # Voxel size calculations
    h_range = hkl_bounds["h_max"] - hkl_bounds["h_min"]
    k_range = hkl_bounds["k_max"] - hkl_bounds["k_min"]
    l_range = hkl_bounds["l_max"] - hkl_bounds["l_min"]

    voxel_size_h = h_range / ndiv_h
    voxel_size_k = k_range / ndiv_k
    voxel_size_l = l_range / ndiv_l

    summary_text.append("")
    summary_text.append("Voxel Sizes (in reciprocal lattice units):")
    summary_text.append(f"  ΔH: {voxel_size_h:.4f}")
    summary_text.append(f"  ΔK: {voxel_size_k:.4f}")
    summary_text.append(f"  ΔL: {voxel_size_l:.4f}")

    # Save text summary
    summary_file = output_dir / "grid_summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_text))
    logger.info(f"Saved grid summary to {summary_file}")

    # Generate conceptual visualization
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "3d"})

    # Draw wireframe box representing grid bounds
    h_min, h_max = hkl_bounds["h_min"], hkl_bounds["h_max"]
    k_min, k_max = hkl_bounds["k_min"], hkl_bounds["k_max"]
    l_min, l_max = hkl_bounds["l_min"], hkl_bounds["l_max"]

    # Define box corners
    corners = [
        [h_min, k_min, l_min],
        [h_max, k_min, l_min],
        [h_max, k_max, l_min],
        [h_min, k_max, l_min],
        [h_min, k_min, l_max],
        [h_max, k_min, l_max],
        [h_max, k_max, l_max],
        [h_min, k_max, l_max],
    ]

    # Draw edges
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # bottom face
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # top face
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # vertical edges
    ]

    for edge in edges:
        points = np.array([corners[edge[0]], corners[edge[1]]])
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], "b-", linewidth=2)

    # Add some sample grid points
    n_sample_points = min(1000, total_voxels // 100)
    if n_sample_points > 0:
        sample_h = np.random.uniform(h_min, h_max, n_sample_points)
        sample_k = np.random.uniform(k_min, k_max, n_sample_points)
        sample_l = np.random.uniform(l_min, l_max, n_sample_points)
        ax.scatter(sample_h, sample_k, sample_l, c="red", s=1, alpha=0.3)

    ax.set_xlabel("H")
    ax.set_ylabel("K")
    ax.set_zlabel("L")
    ax.set_title(
        "Global Voxel Grid Conceptual Visualization\n(Blue wireframe shows HKL bounds)"
    )

    # Save conceptual plot
    plot_file = output_dir / "grid_visualization_conceptual.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved grid visualization to {plot_file}")


def generate_voxel_occupancy_plots(
    voxel_data: Dict[str, np.ndarray], grid_def: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    """
    Generate voxel occupancy/redundancy analysis plots.

    Args:
        voxel_data: Voxel data dictionary
        grid_def: Grid definition dictionary
        output_dir: Output directory for plots

    Returns:
        Dictionary with occupancy statistics
    """
    logger.info("Generating voxel occupancy plots")

    num_observations = voxel_data["num_observations"]
    h_center = voxel_data["H_center"]
    k_center = voxel_data["K_center"]
    l_center = voxel_data["L_center"]

    # Calculate occupancy statistics
    stats = {
        "min_observations": int(np.min(num_observations)),
        "max_observations": int(np.max(num_observations)),
        "mean_observations": float(np.mean(num_observations)),
        "median_observations": float(np.median(num_observations)),
        "total_observations": int(np.sum(num_observations)),
        "voxels_with_data": int(np.sum(num_observations > 0)),
        "total_voxels": len(num_observations),
    }

    # Calculate percentage with low redundancy
    low_redundancy_thresh = 3
    stats[f"percent_voxels_lt_{low_redundancy_thresh}"] = float(
        100 * np.sum(num_observations < low_redundancy_thresh) / len(num_observations)
    )

    logger.info(f"Occupancy stats: {stats}")

    # Reshape occupancy into 3D grid for slicing
    ndiv_h, ndiv_k, ndiv_l = grid_def["ndiv_h"], grid_def["ndiv_k"], grid_def["ndiv_l"]
    hkl_bounds = grid_def["hkl_bounds"]

    # Create 3D occupancy grid
    occupancy_grid = np.zeros((ndiv_h, ndiv_k, ndiv_l))

    # Map HKL centers to grid indices
    h_indices = np.round(
        (h_center - hkl_bounds["h_min"])
        / (hkl_bounds["h_max"] - hkl_bounds["h_min"])
        * (ndiv_h - 1)
    ).astype(int)
    k_indices = np.round(
        (k_center - hkl_bounds["k_min"])
        / (hkl_bounds["k_max"] - hkl_bounds["k_min"])
        * (ndiv_k - 1)
    ).astype(int)
    l_indices = np.round(
        (l_center - hkl_bounds["l_min"])
        / (hkl_bounds["l_max"] - hkl_bounds["l_min"])
        * (ndiv_l - 1)
    ).astype(int)

    # Clip indices to valid range
    h_indices = np.clip(h_indices, 0, ndiv_h - 1)
    k_indices = np.clip(k_indices, 0, ndiv_k - 1)
    l_indices = np.clip(l_indices, 0, ndiv_l - 1)

    # Fill occupancy grid
    for i, (hi, ki, li) in enumerate(zip(h_indices, k_indices, l_indices)):
        occupancy_grid[hi, ki, li] = num_observations[i]

    # Generate 2D slice plots
    slice_configs = [
        (2, ndiv_l // 2, "H-K", "H", "K", "voxel_occupancy_slice_L0.png"),
        (1, ndiv_k // 2, "H-L", "H", "L", "voxel_occupancy_slice_K0.png"),
        (0, ndiv_h // 2, "K-L", "K", "L", "voxel_occupancy_slice_H0.png"),
    ]

    for slice_dim, slice_idx, title, xlabel, ylabel, filename in slice_configs:
        plot_3d_grid_slice(
            grid_data_3d=occupancy_grid,
            slice_dim_idx=slice_dim,
            slice_val_idx=slice_idx,
            title=f"Voxel Occupancy - {title} Slice",
            output_path=str(output_dir / filename),
            cmap="viridis",
            norm=None,
            xlabel=xlabel,
            ylabel=ylabel,
            aspect="auto",
        )

    # Generate occupancy histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use log bins for better visualization
    max_obs = max(stats["max_observations"], 1)
    bins = np.logspace(0, np.log10(max_obs + 1), 50)

    counts, _, _ = ax.hist(
        num_observations[num_observations > 0],
        bins=bins,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Number of Observations per Voxel")
    ax.set_ylabel("Number of Voxels")
    ax.set_title("Voxel Occupancy Distribution")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Mean: {stats['mean_observations']:.1f}\n"
    stats_text += f"Median: {stats['median_observations']:.1f}\n"
    stats_text += f"Min: {stats['min_observations']}\n"
    stats_text += f"Max: {stats['max_observations']}"

    ax.text(
        0.75,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    histogram_file = output_dir / "voxel_occupancy_histogram.png"
    fig.savefig(histogram_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved occupancy histogram to {histogram_file}")

    return stats


def generate_scaling_parameter_plots(
    scaling_params: Dict[str, Any], output_dir: Path
) -> None:
    """
    Generate plots for refined scaling model parameters.

    Args:
        scaling_params: Scaling parameters dictionary
        output_dir: Output directory for plots
    """
    logger.info("Generating scaling parameter plots")

    refined_params = scaling_params["refined_parameters"]

    # Extract per-still scales
    still_ids = list(refined_params.keys())
    scales = [params["multiplicative_scale"] for params in refined_params.values()]
    still_indices = range(len(still_ids))

    # Plot per-still scales
    plot_parameter_vs_index(
        param_values=scales,
        index_values=list(still_indices),
        title="Per-Still Multiplicative Scale Factors",
        param_label="Multiplicative Scale (b_i)",
        index_label="Still Index",
        output_path=str(output_dir / "scaling_params_b_i.png"),
    )

    # Plot resolution smoother if enabled
    if "resolution_smoother" in scaling_params and scaling_params[
        "resolution_smoother"
    ].get("enabled", False):
        control_points = scaling_params["resolution_smoother"].get("control_points", [])

        if control_points:
            # Create a simple evaluation function for the smoother
            def smoother_eval_func(q_values):
                # Simple linear interpolation between control points
                q_control = np.linspace(0.1, 2.0, len(control_points))
                return np.interp(q_values, q_control, control_points)

            # Plot smoother curve
            plot_smoother_curve(
                smoother_eval_func=smoother_eval_func,
                x_range=(0.1, 2.0),
                num_points=100,
                title="Resolution Smoother Function a(|q|)",
                output_path=str(output_dir / "scaling_resolution_smoother.png"),
                control_points_x=np.linspace(0.1, 2.0, len(control_points)),
                control_points_y=control_points,
            )

    # Generate parameter summary text
    summary_text = []
    summary_text.append("=== Scaling Model Parameters Summary ===\n")

    summary_text.append(f"Number of Stills: {len(refined_params)}")
    summary_text.append("Scale Factor Statistics:")
    summary_text.append(f"  Mean: {np.mean(scales):.4f}")
    summary_text.append(f"  Std Dev: {np.std(scales):.4f}")
    summary_text.append(f"  Min: {np.min(scales):.4f}")
    summary_text.append(f"  Max: {np.max(scales):.4f}")
    summary_text.append("")

    # Refinement statistics
    if "refinement_statistics" in scaling_params:
        ref_stats = scaling_params["refinement_statistics"]
        summary_text.append("Refinement Statistics:")
        summary_text.append(f"  Iterations: {ref_stats.get('n_iterations', 'N/A')}")
        summary_text.append(
            f"  Final R-factor: {ref_stats.get('final_r_factor', 'N/A')}"
        )
        summary_text.append(
            f"  Converged: {ref_stats.get('convergence_achieved', 'N/A')}"
        )

        if "parameter_shifts" in ref_stats:
            summary_text.append("  Final Parameter Shifts:")
            for param, shift in ref_stats["parameter_shifts"].items():
                summary_text.append(f"    {param}: {shift}")

    # Resolution smoother info
    if "resolution_smoother" in scaling_params:
        res_smooth = scaling_params["resolution_smoother"]
        summary_text.append("")
        summary_text.append(
            f"Resolution Smoother: {'Enabled' if res_smooth.get('enabled', False) else 'Disabled'}"
        )
        if res_smooth.get("enabled", False):
            control_points = res_smooth.get("control_points", [])
            summary_text.append(f"  Control Points: {len(control_points)}")

    # Save parameter summary
    params_summary_file = output_dir / "scaling_parameters_summary.txt"
    with open(params_summary_file, "w") as f:
        f.write("\n".join(summary_text))
    logger.info(f"Saved scaling parameters summary to {params_summary_file}")


def generate_merged_voxel_plots(
    voxel_data: Dict[str, np.ndarray],
    grid_def: Dict[str, Any],
    output_dir: Path,
    max_plot_points: int,
) -> None:
    """
    Generate plots for merged voxel data visualization.

    Args:
        voxel_data: Voxel data dictionary
        grid_def: Grid definition dictionary
        output_dir: Output directory for plots
        max_plot_points: Maximum points for scatter plots
    """
    logger.info("Generating merged voxel data plots")

    intensities = voxel_data["I_merged_relative"]
    sigmas = voxel_data["Sigma_merged_relative"]
    q_magnitudes = voxel_data["q_magnitude_center"]
    h_center = voxel_data["H_center"]
    k_center = voxel_data["K_center"]
    l_center = voxel_data["L_center"]

    # Filter out zero/negative intensities for log plots
    positive_mask = intensities > 0

    # Calculate I/sigma
    i_sig_ratio = np.where(sigmas > 0, intensities / sigmas, 0)

    # Reshape intensity and sigma into 3D grids for slicing
    ndiv_h, ndiv_k, ndiv_l = grid_def["ndiv_h"], grid_def["ndiv_k"], grid_def["ndiv_l"]
    hkl_bounds = grid_def["hkl_bounds"]

    # Create 3D grids
    intensity_grid = np.zeros((ndiv_h, ndiv_k, ndiv_l))
    sigma_grid = np.zeros((ndiv_h, ndiv_k, ndiv_l))
    isigi_grid = np.zeros((ndiv_h, ndiv_k, ndiv_l))

    # Map HKL centers to grid indices
    h_indices = np.round(
        (h_center - hkl_bounds["h_min"])
        / (hkl_bounds["h_max"] - hkl_bounds["h_min"])
        * (ndiv_h - 1)
    ).astype(int)
    k_indices = np.round(
        (k_center - hkl_bounds["k_min"])
        / (hkl_bounds["k_max"] - hkl_bounds["k_min"])
        * (ndiv_k - 1)
    ).astype(int)
    l_indices = np.round(
        (l_center - hkl_bounds["l_min"])
        / (hkl_bounds["l_max"] - hkl_bounds["l_min"])
        * (ndiv_l - 1)
    ).astype(int)

    # Clip indices to valid range
    h_indices = np.clip(h_indices, 0, ndiv_h - 1)
    k_indices = np.clip(k_indices, 0, ndiv_k - 1)
    l_indices = np.clip(l_indices, 0, ndiv_l - 1)

    # Fill grids
    for i, (hi, ki, li) in enumerate(zip(h_indices, k_indices, l_indices)):
        intensity_grid[hi, ki, li] = intensities[i]
        sigma_grid[hi, ki, li] = sigmas[i]
        isigi_grid[hi, ki, li] = i_sig_ratio[i]

    # Generate intensity slice plots (log scale)
    intensity_slice_configs = [
        (2, ndiv_l // 2, "H-K", "H", "K", "merged_intensity_slice_L0.png"),
        (1, ndiv_k // 2, "H-L", "H", "L", "merged_intensity_slice_K0.png"),
        (0, ndiv_h // 2, "K-L", "K", "L", "merged_intensity_slice_H0.png"),
    ]

    # Create log normalization for intensity plots
    from matplotlib.colors import LogNorm

    positive_intensities = intensities[intensities > 0]
    if len(positive_intensities) > 0:
        log_norm = LogNorm(
            vmin=positive_intensities.min(), vmax=positive_intensities.max()
        )
    else:
        log_norm = None

    for (
        slice_dim,
        slice_idx,
        title,
        xlabel,
        ylabel,
        filename,
    ) in intensity_slice_configs:
        plot_3d_grid_slice(
            grid_data_3d=intensity_grid,
            slice_dim_idx=slice_dim,
            slice_val_idx=slice_idx,
            title=f"Merged Intensity - {title} Slice (Log Scale)",
            output_path=str(output_dir / filename),
            cmap="viridis",
            norm=log_norm,
            xlabel=xlabel,
            ylabel=ylabel,
            aspect="auto",
        )

    # Generate sigma slice plots
    sigma_slice_configs = [
        (2, ndiv_l // 2, "H-K", "H", "K", "merged_sigma_slice_L0.png"),
        (1, ndiv_k // 2, "H-L", "H", "L", "merged_sigma_slice_K0.png"),
        (0, ndiv_h // 2, "K-L", "K", "L", "merged_sigma_slice_H0.png"),
    ]

    for slice_dim, slice_idx, title, xlabel, ylabel, filename in sigma_slice_configs:
        plot_3d_grid_slice(
            grid_data_3d=sigma_grid,
            slice_dim_idx=slice_dim,
            slice_val_idx=slice_idx,
            title=f"Merged Sigma - {title} Slice",
            output_path=str(output_dir / filename),
            cmap="plasma",
            norm=None,
            xlabel=xlabel,
            ylabel=ylabel,
            aspect="auto",
        )

    # Generate I/sigma slice plots
    isigi_slice_configs = [
        (2, ndiv_l // 2, "H-K", "H", "K", "merged_isigi_slice_L0.png"),
        (1, ndiv_k // 2, "H-L", "H", "L", "merged_isigi_slice_K0.png"),
        (0, ndiv_h // 2, "K-L", "K", "L", "merged_isigi_slice_H0.png"),
    ]

    for slice_dim, slice_idx, title, xlabel, ylabel, filename in isigi_slice_configs:
        plot_3d_grid_slice(
            grid_data_3d=isigi_grid,
            slice_dim_idx=slice_dim,
            slice_val_idx=slice_idx,
            title=f"Merged I/σ - {title} Slice",
            output_path=str(output_dir / filename),
            cmap="coolwarm",
            norm=None,
            xlabel=xlabel,
            ylabel=ylabel,
            aspect="auto",
        )

    # Generate radial average plot
    mask_for_radial = positive_mask & (q_magnitudes > 0)
    if np.sum(mask_for_radial) > 0:
        plot_radial_average(
            q_magnitudes=q_magnitudes[mask_for_radial],
            intensities=intensities[mask_for_radial],
            num_bins=50,
            title="Radial Average of Merged Intensities",
            output_path=str(output_dir / "merged_radial_average.png"),
            sigmas=sigmas[mask_for_radial],
        )

    # Generate intensity histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    positive_intensities = intensities[positive_mask]
    if len(positive_intensities) > 0:
        # Use log bins for better visualization
        bins = np.logspace(
            np.log10(positive_intensities.min()),
            np.log10(positive_intensities.max()),
            50,
        )

        ax.hist(
            positive_intensities, bins=bins, alpha=0.7, edgecolor="black", linewidth=0.5
        )
        ax.set_xscale("log")
    else:
        ax.hist(intensities, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Merged Intensity")
    ax.set_ylabel("Number of Voxels")
    ax.set_title("Distribution of Merged Intensities")
    ax.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"Total Voxels: {len(intensities)}\n"
    stats_text += f"Positive Intensities: {np.sum(positive_mask)}\n"
    stats_text += (
        f"Mean (positive): {np.mean(positive_intensities):.2f}\n"
        if len(positive_intensities) > 0
        else "Mean: N/A\n"
    )
    stats_text += (
        f"Median (positive): {np.median(positive_intensities):.2f}"
        if len(positive_intensities) > 0
        else "Median: N/A"
    )

    ax.text(
        0.75,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    histogram_file = output_dir / "merged_intensity_histogram.png"
    fig.savefig(histogram_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved intensity histogram to {histogram_file}")


def generate_comprehensive_summary(
    grid_def: Dict[str, Any],
    scaling_params: Dict[str, Any],
    voxel_data: Dict[str, np.ndarray],
    occupancy_stats: Dict[str, Any],
    input_files: Dict[str, str],
    output_dir: Path,
) -> None:
    """
    Generate comprehensive summary report.

    Args:
        grid_def: Grid definition dictionary
        scaling_params: Scaling parameters dictionary
        voxel_data: Voxel data dictionary
        occupancy_stats: Occupancy statistics
        input_files: Input file paths
        output_dir: Output directory
    """
    logger.info("Generating comprehensive summary report")

    summary_text = []
    summary_text.append("=== Phase 3 Diagnostics Summary Report ===")
    summary_text.append(
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    summary_text.append("")

    # Input files section
    summary_text.append("Input Files:")
    for file_type, file_path in input_files.items():
        summary_text.append(f"  {file_type}: {file_path}")
    summary_text.append("")

    # Grid definition summary
    summary_text.append("Global Voxel Grid:")
    crystal_info = grid_def["crystal_avg_ref"]
    if "unit_cell_params" in crystal_info:
        uc_params = crystal_info["unit_cell_params"]
        summary_text.append(
            f"  Unit Cell: {uc_params[0]:.2f} {uc_params[1]:.2f} {uc_params[2]:.2f} {uc_params[3]:.1f} {uc_params[4]:.1f} {uc_params[5]:.1f}"
        )
    if "space_group" in crystal_info:
        summary_text.append(f"  Space Group: {crystal_info['space_group']}")

    hkl_bounds = grid_def["hkl_bounds"]
    summary_text.append(
        f"  HKL Range: H({hkl_bounds['h_min']}:{hkl_bounds['h_max']}) K({hkl_bounds['k_min']}:{hkl_bounds['k_max']}) L({hkl_bounds['l_min']}:{hkl_bounds['l_max']})"
    )
    summary_text.append(
        f"  Divisions: {grid_def['ndiv_h']} × {grid_def['ndiv_k']} × {grid_def['ndiv_l']}"
    )
    summary_text.append(f"  Total Voxels: {grid_def['total_voxels']:,}")
    summary_text.append("")

    # Occupancy statistics
    summary_text.append("Voxel Occupancy:")
    summary_text.append(
        f"  Voxels with Data: {occupancy_stats['voxels_with_data']:,} / {occupancy_stats['total_voxels']:,} ({100*occupancy_stats['voxels_with_data']/occupancy_stats['total_voxels']:.1f}%)"
    )
    summary_text.append(
        f"  Total Observations: {occupancy_stats['total_observations']:,}"
    )
    summary_text.append(
        f"  Mean Observations per Voxel: {occupancy_stats['mean_observations']:.1f}"
    )
    summary_text.append(
        f"  Median Observations per Voxel: {occupancy_stats['median_observations']:.1f}"
    )
    summary_text.append(
        f"  Range: {occupancy_stats['min_observations']} - {occupancy_stats['max_observations']}"
    )
    summary_text.append(
        f"  Voxels with < 3 observations: {occupancy_stats['percent_voxels_lt_3']:.1f}%"
    )
    summary_text.append("")

    # Scaling model summary
    refined_params = scaling_params["refined_parameters"]
    scales = [params["multiplicative_scale"] for params in refined_params.values()]

    summary_text.append("Relative Scaling:")
    summary_text.append(f"  Number of Stills: {len(refined_params)}")
    summary_text.append(
        f"  Scale Factor Range: {np.min(scales):.4f} - {np.max(scales):.4f}"
    )
    summary_text.append(
        f"  Scale Factor Mean ± Std: {np.mean(scales):.4f} ± {np.std(scales):.4f}"
    )

    if "refinement_statistics" in scaling_params:
        ref_stats = scaling_params["refinement_statistics"]
        summary_text.append(
            f"  Refinement Iterations: {ref_stats.get('n_iterations', 'N/A')}"
        )
        summary_text.append(
            f"  Final R-factor: {ref_stats.get('final_r_factor', 'N/A'):.4f}"
            if isinstance(ref_stats.get("final_r_factor"), (int, float))
            else f"  Final R-factor: {ref_stats.get('final_r_factor', 'N/A')}"
        )
        summary_text.append(
            f"  Convergence: {'Yes' if ref_stats.get('convergence_achieved', False) else 'No'}"
        )

    if "resolution_smoother" in scaling_params:
        res_smooth = scaling_params["resolution_smoother"]
        summary_text.append(
            f"  Resolution Smoother: {'Enabled' if res_smooth.get('enabled', False) else 'Disabled'}"
        )
    summary_text.append("")

    # Merged intensity statistics
    intensities = voxel_data["I_merged_relative"]
    sigmas = voxel_data["Sigma_merged_relative"]
    positive_mask = intensities > 0

    summary_text.append("Merged Intensities:")
    summary_text.append(f"  Total Voxels: {len(intensities):,}")
    summary_text.append(
        f"  Voxels with Positive Intensity: {np.sum(positive_mask):,} ({100*np.sum(positive_mask)/len(intensities):.1f}%)"
    )

    if np.sum(positive_mask) > 0:
        positive_intensities = intensities[positive_mask]
        summary_text.append(
            f"  Intensity Range (positive): {np.min(positive_intensities):.2e} - {np.max(positive_intensities):.2e}"
        )
        summary_text.append(
            f"  Intensity Mean ± Std (positive): {np.mean(positive_intensities):.2e} ± {np.std(positive_intensities):.2e}"
        )

        # I/sigma statistics
        positive_sigmas = sigmas[positive_mask]
        valid_sigma_mask = positive_sigmas > 0
        if np.sum(valid_sigma_mask) > 0:
            i_sig_ratios = (
                positive_intensities[valid_sigma_mask]
                / positive_sigmas[valid_sigma_mask]
            )
            summary_text.append(
                f"  I/σ Mean ± Std: {np.mean(i_sig_ratios):.2f} ± {np.std(i_sig_ratios):.2f}"
            )
            summary_text.append(
                f"  I/σ Range: {np.min(i_sig_ratios):.2f} - {np.max(i_sig_ratios):.2f}"
            )

    summary_text.append("")

    # Resolution statistics
    q_magnitudes = voxel_data["q_magnitude_center"]
    summary_text.append("Resolution Coverage:")
    summary_text.append(
        f"  Q Range: {np.min(q_magnitudes):.3f} - {np.max(q_magnitudes):.3f} Å⁻¹"
    )

    # Convert to d-spacing
    d_spacings = 2 * np.pi / q_magnitudes[q_magnitudes > 0]
    if len(d_spacings) > 0:
        summary_text.append(
            f"  d-spacing Range: {np.min(d_spacings):.2f} - {np.max(d_spacings):.2f} Å"
        )

    summary_text.append("")

    # Generated plots summary
    summary_text.append("Generated Diagnostic Plots:")
    plot_files = [
        "grid_summary.txt",
        "grid_visualization_conceptual.png",
        "voxel_occupancy_slice_L0.png",
        "voxel_occupancy_histogram.png",
        "scaling_params_b_i.png",
        "merged_intensity_slice_L0.png",
        "merged_radial_average.png",
        "merged_intensity_histogram.png",
    ]

    for plot_file in plot_files:
        if (output_dir / plot_file).exists():
            summary_text.append(f"  ✓ {plot_file}")
        else:
            summary_text.append(f"  ✗ {plot_file} (missing)")

    # Save comprehensive summary
    summary_file = output_dir / "phase3_diagnostics_summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_text))
    logger.info(f"Saved comprehensive summary to {summary_file}")


def main():
    """Main function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)

    logger.info("Starting Phase 3 outputs visual diagnostics")
    logger.info(f"Grid definition file: {args.grid_definition_file}")
    logger.info(f"Scaling parameters file: {args.scaling_model_params_file}")
    logger.info(f"Voxel data file: {args.voxel_data_file}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Validate input files
        validate_input_files(args)

        # Ensure output directory exists
        output_dir = ensure_output_dir(args.output_dir)

        # Load input data
        logger.info("Loading input data...")
        grid_def = load_grid_definition(args.grid_definition_file)
        scaling_params = load_scaling_parameters(args.scaling_model_params_file)
        voxel_data = load_voxel_data(args.voxel_data_file)

        # Generate diagnostics
        logger.info("Generating diagnostics...")

        # Grid summary
        generate_grid_summary(grid_def, output_dir)

        # Voxel occupancy analysis
        occupancy_stats = generate_voxel_occupancy_plots(
            voxel_data, grid_def, output_dir
        )

        # Scaling parameter plots
        generate_scaling_parameter_plots(scaling_params, output_dir)

        # Merged voxel data visualization
        generate_merged_voxel_plots(
            voxel_data, grid_def, output_dir, args.max_plot_points
        )

        # Comprehensive summary
        input_files = {
            "Grid Definition": args.grid_definition_file,
            "Scaling Parameters": args.scaling_model_params_file,
            "Voxel Data": args.voxel_data_file,
        }
        if args.experiments_list_file:
            input_files["Experiments List"] = args.experiments_list_file
        if args.corrected_pixel_data_dir:
            input_files["Pixel Data Directories"] = args.corrected_pixel_data_dir

        # Use standard library for timestamp

        generate_comprehensive_summary(
            grid_def,
            scaling_params,
            voxel_data,
            occupancy_stats,
            input_files,
            output_dir,
        )

        # Clean up
        close_all_figures()

        logger.info("=== Phase 3 Visual Diagnostics Completed Successfully ===")
        logger.info(f"All diagnostic plots and reports saved to: {output_dir}")

        # List generated files
        logger.info("Generated files:")
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                logger.info(f"  {file_path.name}")

    except Exception as e:
        logger.error(f"Phase 3 diagnostics failed: {e}")
        logger.error("Check the log for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()
