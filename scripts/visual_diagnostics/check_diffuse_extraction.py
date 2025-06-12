#!/usr/bin/env python3
"""
Visual diagnostics for diffuse scattering extraction verification.

This script takes outputs from Phase 1 and Phase 2 (DataExtractor NPZ file) and generates
a series of diagnostic plots to visually verify the correctness of the diffuse scattering
extraction and correction process.

Key diagnostic plots include:
1. Raw image with extracted diffuse pixels overlay
2. Intensity correction effects (simplified version)
3. Q-space coverage projections
4. Radial Q-space distribution
5. Intensity distribution histogram
6. Intensity heatmap (conditional on pixel coordinates)
7. Sigma vs intensity scatter plot
8. I/sigma histogram

Author: DiffusePipe
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib

matplotlib.use("Agg")  # Set early for non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Default maximum number of points to display in scatter plots for performance
DEFAULT_MAX_SCATTER_POINTS = 100000

# Import DIALS/DXTBX libraries
try:
    from dxtbx.imageset import ImageSetFactory
    from dxtbx.model.experiment_list import ExperimentListFactory
except ImportError as e:
    print(f"Error importing DIALS/DXTBX libraries: {e}")
    print("Please ensure DIALS is properly installed and accessible.")
    sys.exit(1)

# Import project utilities
try:
    from plot_utils import (
        plot_detector_image,
        plot_spot_overlay,
        setup_logging_for_plots,
        ensure_output_dir,
        close_all_figures,
    )
except ImportError as e:
    print(f"Error importing plot utilities: {e}")
    print("Please ensure you're running from the scripts/visual_diagnostics directory.")
    sys.exit(1)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visual diagnostics for diffuse scattering extraction verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python check_diffuse_extraction.py \\
    --raw-image ../../747/lys_nitr_10_6_0491.cbf \\
    --expt ../../lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt \\
    --total-mask ../../lys_nitr_10_6_0491_dials_processing/total_diffuse_mask.pickle \\
    --npz-file extraction_output.npz

  # With optional masks and background map
  python check_diffuse_extraction.py \\
    --raw-image image.cbf --expt experiment.expt \\
    --total-mask total_mask.pickle --npz-file data.npz \\
    --bragg-mask bragg_mask.pickle \\
    --pixel-mask pixel_mask.pickle \\
    --bg-map background_map.npy \\
    --output-dir custom_output \\
    --verbose
        """,
    )

    # Required arguments
    parser.add_argument(
        "--raw-image", type=str, required=True, help="Path to raw CBF image file"
    )
    parser.add_argument(
        "--expt", type=str, required=True, help="Path to DIALS experiment .expt file"
    )
    parser.add_argument(
        "--total-mask",
        type=str,
        required=True,
        help="Path to total diffuse mask .pickle file",
    )
    parser.add_argument(
        "--npz-file",
        type=str,
        required=True,
        help="Path to DataExtractor output .npz file",
    )

    # Optional arguments
    parser.add_argument(
        "--bragg-mask", type=str, help="Path to Bragg mask .pickle file (optional)"
    )
    parser.add_argument(
        "--pixel-mask",
        type=str,
        help="Path to global pixel mask .pickle file (optional)",
    )
    parser.add_argument(
        "--bg-map",
        type=str,
        help="Path to measured background map .npy/.pickle file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extraction_visual_check",
        help="Output directory for diagnostic plots (default: extraction_visual_check)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=DEFAULT_MAX_SCATTER_POINTS,
        help="Maximum number of points to display in scatter plots for performance.",
    )

    return parser.parse_args()


def load_npz_data(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Load NPZ data from DataExtractor output.

    Args:
        npz_path: Path to NPZ file

    Returns:
        Dictionary containing loaded arrays
    """
    try:
        data = np.load(npz_path)
        result = {}

        # Required arrays
        required_keys = ["q_vectors", "intensities", "sigmas"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Required key '{key}' not found in NPZ file")
            result[key] = data[key]

        # Optional coordinate arrays
        optional_keys = [
            "original_panel_ids",
            "original_fast_coords",
            "original_slow_coords",
        ]
        missing_coords = []
        for key in optional_keys:
            if key in data:
                result[key] = data[key]
            else:
                missing_coords.append(key)

        if missing_coords:
            logger.warning(
                f"Missing coordinate arrays in NPZ file: {missing_coords}. "
                "Some plots (pixel overlay, intensity heatmap) will be limited or skipped."
            )

        logger.info(f"Loaded NPZ data with {len(result['q_vectors'])} diffuse points")
        return result

    except Exception as e:
        logger.error(f"Failed to load NPZ data from {npz_path}: {e}")
        raise


def load_mask_pickle(mask_path: str) -> Optional[Tuple]:
    """
    Load a mask pickle file.

    Args:
        mask_path: Path to pickle file

    Returns:
        Mask tuple or None on error
    """
    try:
        with open(mask_path, "rb") as f:
            mask_data = pickle.load(f)
        logger.info(f"Loaded mask from {mask_path}")
        return mask_data
    except Exception as e:
        logger.error(f"Failed to load mask from {mask_path}: {e}")
        return None


def load_cbf_image(image_path: str):
    """
    Load CBF image using DXTBX.

    Args:
        image_path: Path to CBF file

    Returns:
        ImageSet object
    """
    try:
        imageset = ImageSetFactory.new([image_path])
        logger.info(f"Loaded CBF image from {image_path}")
        return imageset
    except Exception as e:
        logger.error(f"Failed to load CBF image from {image_path}: {e}")
        raise


def load_experiment(expt_path: str):
    """
    Load DIALS experiment.

    Args:
        expt_path: Path to .expt file

    Returns:
        First Experiment object from the list
    """
    try:
        experiments = ExperimentListFactory.from_json_file(expt_path)
        if len(experiments) == 0:
            raise ValueError("No experiments found in file")

        experiment = experiments[0]
        logger.info(f"Loaded experiment from {expt_path}")
        return experiment
    except Exception as e:
        logger.error(f"Failed to load experiment from {expt_path}: {e}")
        raise


def load_background_map(bg_map_path: str) -> Optional[np.ndarray]:
    """
    Load background map from .npy or .pickle file.

    Args:
        bg_map_path: Path to background map file

    Returns:
        Background map array or None on error
    """
    try:
        if bg_map_path.endswith(".npy"):
            bg_map = np.load(bg_map_path)
        elif bg_map_path.endswith(".pickle"):
            with open(bg_map_path, "rb") as f:
                bg_map = pickle.load(f)
        else:
            raise ValueError("Background map must be .npy or .pickle file")

        logger.info(f"Loaded background map from {bg_map_path}")
        return bg_map
    except Exception as e:
        logger.error(f"Failed to load background map from {bg_map_path}: {e}")
        return None


def validate_input_files(args: argparse.Namespace) -> None:
    """Validate that all required input files exist."""
    required_files = [args.raw_image, args.expt, args.total_mask, args.npz_file]

    optional_files = [args.bragg_mask, args.pixel_mask, args.bg_map]

    # Check required files
    for file_path in required_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Check optional files
    for file_path in optional_files:
        if file_path and not Path(file_path).exists():
            logger.warning(f"Optional file not found: {file_path}")


def plot_diffuse_pixel_overlay(
    raw_image_data: Union[Any, np.ndarray],
    panel_id: int,
    pixel_coords_for_panel: List[Tuple[float, float]],
    title: str,
    output_path: str,
    bragg_coords: Optional[List[Tuple[float, float]]] = None,
    pixel_mask_coords: Optional[List[Tuple[float, float]]] = None,
) -> plt.Figure:
    """
    Plot raw image with extracted diffuse pixels overlaid.

    Args:
        raw_image_data: Raw detector image data
        panel_id: Panel ID being plotted
        pixel_coords_for_panel: List of (fast, slow) coordinates for diffuse pixels
        title: Plot title
        output_path: Output file path
        bragg_coords: Optional Bragg pixel coordinates
        pixel_mask_coords: Optional masked pixel coordinates

    Returns:
        matplotlib Figure object
    """
    fig = plot_spot_overlay(
        raw_image_data,
        pixel_coords_for_panel,
        title=title,
        output_path=output_path,
        spot_color="green",
        spot_size=1,
        log_scale=True,
        max_points=None,  # Subsampling handled before calling this function
    )

    # Add additional overlays if provided
    if bragg_coords or pixel_mask_coords:
        ax = fig.axes[0]

        if bragg_coords:
            bragg_x, bragg_y = zip(*bragg_coords) if bragg_coords else ([], [])
            ax.scatter(bragg_x, bragg_y, c="red", s=1, alpha=0.5, label="Bragg regions")

        if pixel_mask_coords:
            mask_x, mask_y = zip(*pixel_mask_coords) if pixel_mask_coords else ([], [])
            ax.scatter(mask_x, mask_y, c="black", s=1, alpha=0.3, label="Masked pixels")

        ax.legend()

        # Re-save with updated legend
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_q_projections(
    q_vectors: np.ndarray,
    intensities: np.ndarray,
    output_dir: Path,
    max_points: Optional[int] = DEFAULT_MAX_SCATTER_POINTS,
) -> None:
    """
    Generate Q-space projection plots.

    Args:
        q_vectors: Array of shape (N, 3) with qx, qy, qz
        intensities: Array of shape (N,) with intensity values
        output_dir: Output directory for plots
        max_points: Maximum number of points to plot for performance
    """
    # Apply subsampling if necessary
    if max_points and len(q_vectors) > max_points:
        indices = np.random.choice(len(q_vectors), max_points, replace=False)
        q_subset = q_vectors[indices]
        int_subset = intensities[indices]
        sampled_note = f" (sampled {max_points} of {len(q_vectors)} points)"
    else:
        q_subset = q_vectors
        int_subset = intensities
        sampled_note = ""

    qx, qy, qz = q_subset[:, 0], q_subset[:, 1], q_subset[:, 2]

    # Create three projection plots
    projections = [
        (qx, qy, "qx (Å⁻¹)", "qy (Å⁻¹)", "qx_qy"),
        (qx, qz, "qx (Å⁻¹)", "qz (Å⁻¹)", "qx_qz"),
        (qy, qz, "qy (Å⁻¹)", "qz (Å⁻¹)", "qy_qz"),
    ]

    for x_data, y_data, xlabel, ylabel, filename in projections:
        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            x_data, y_data, c=int_subset, cmap="viridis", alpha=0.6, s=1
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Q-space projection: {xlabel} vs {ylabel}{sampled_note}")

        plt.colorbar(scatter, ax=ax, label="Intensity")

        output_path = output_dir / f"q_projection_{filename}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved Q-space projection to {output_path}")


def plot_radial_q(
    q_vectors: np.ndarray,
    intensities: np.ndarray,
    output_path: Path,
    max_points: Optional[int] = DEFAULT_MAX_SCATTER_POINTS,
) -> None:
    """
    Plot intensity vs radial Q.

    Args:
        q_vectors: Array of shape (N, 3) with qx, qy, qz
        intensities: Array of shape (N,) with intensity values
        output_path: Output file path
        max_points: Maximum number of points to plot for performance
    """
    # Apply subsampling if necessary
    if max_points and len(q_vectors) > max_points:
        indices = np.random.choice(len(q_vectors), max_points, replace=False)
        q_subset = q_vectors[indices]
        int_subset = intensities[indices]
        sampled_note = f" (sampled {max_points} of {len(q_vectors)} points)"
    else:
        q_subset = q_vectors
        int_subset = intensities
        sampled_note = ""

    q_radial = np.sqrt(np.sum(q_subset**2, axis=1))

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        q_radial, int_subset, c=int_subset, cmap="viridis", alpha=0.6, s=1
    )

    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Intensity vs Radial Q{sampled_note}")

    plt.colorbar(scatter, ax=ax, label="Intensity")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved radial Q plot to {output_path}")


def plot_intensity_histogram(intensities: np.ndarray, output_path: Path) -> None:
    """
    Plot intensity distribution histogram.

    Args:
        intensities: Array of intensity values
        output_path: Output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale histogram
    ax1.hist(intensities, bins=50, alpha=0.7, color="blue", edgecolor="black")
    ax1.set_xlabel("Intensity")
    ax1.set_ylabel("Count")
    ax1.set_title("Intensity Distribution (Linear)")

    # Log scale histogram (for better visualization of tails)
    ax2.hist(intensities, bins=50, alpha=0.7, color="green", edgecolor="black")
    ax2.set_xlabel("Intensity")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")
    ax2.set_title("Intensity Distribution (Log Scale)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved intensity histogram to {output_path}")


def plot_intensity_heatmap(
    panel_id: int,
    pixel_coords_for_panel: List[Tuple[float, float]],
    intensities_for_panel: np.ndarray,
    panel_shape: Tuple[int, int],
    title: str,
    output_path: Path,
    max_points: Optional[int] = 500000,
) -> None:
    """
    Plot intensity heatmap for a detector panel.

    Args:
        panel_id: Panel ID
        pixel_coords_for_panel: List of (fast, slow) coordinates
        intensities_for_panel: Intensity values for each coordinate
        panel_shape: (height, width) of detector panel
        title: Plot title
        output_path: Output file path
        max_points: Maximum number of points to use for heatmap
    """
    # Apply subsampling if necessary
    if max_points and len(pixel_coords_for_panel) > max_points:
        indices = np.random.choice(
            len(pixel_coords_for_panel), max_points, replace=False
        )
        coords_subset = [pixel_coords_for_panel[i] for i in indices]
        intensities_subset = intensities_for_panel[indices]
        sampled_note = (
            f" (sampled {max_points} of {len(pixel_coords_for_panel)} points)"
        )
        title = title + sampled_note
    else:
        coords_subset = pixel_coords_for_panel
        intensities_subset = intensities_for_panel

    # Create 2D array for heatmap
    height, width = panel_shape
    heatmap = np.full((height, width), np.nan)

    # Populate heatmap with intensity values
    for (fast, slow), intensity in zip(coords_subset, intensities_subset):
        if 0 <= slow < height and 0 <= fast < width:
            heatmap[int(slow), int(fast)] = intensity

    fig = plot_detector_image(
        heatmap,
        title=title,
        output_path=str(output_path),
        log_scale=False,
        cmap="viridis",
    )

    plt.close(fig)
    logger.info(f"Saved intensity heatmap to {output_path}")


def plot_sigma_vs_intensity(
    intensities: np.ndarray,
    sigmas: np.ndarray,
    output_path: Path,
    max_points: Optional[int] = DEFAULT_MAX_SCATTER_POINTS,
) -> None:
    """
    Plot sigma vs intensity scatter plot.

    Args:
        intensities: Array of intensity values
        sigmas: Array of sigma (error) values
        output_path: Output file path
        max_points: Maximum number of points to plot for performance
    """
    # Apply subsampling if necessary
    if max_points and len(intensities) > max_points:
        indices = np.random.choice(len(intensities), max_points, replace=False)
        int_subset = intensities[indices]
        sig_subset = sigmas[indices]
        sampled_note = f" (sampled {max_points} of {len(intensities)} points)"
    else:
        int_subset = intensities
        sig_subset = sigmas
        sampled_note = ""

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(int_subset, sig_subset, alpha=0.6, s=1, c="blue")

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Sigma")
    ax.set_title(f"Sigma vs Intensity{sampled_note}")

    # Add ideal sigma relationship line (if reasonable)
    if len(int_subset) > 0:
        max_intensity = np.max(int_subset)
        x_ideal = np.linspace(0, max_intensity, 100)
        y_ideal = np.sqrt(x_ideal)  # Poisson noise relationship
        ax.plot(x_ideal, y_ideal, "r--", alpha=0.5, label="Poisson √I")
        ax.legend()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved sigma vs intensity plot to {output_path}")


def plot_isigi_histogram(
    intensities: np.ndarray, sigmas: np.ndarray, output_path: Path
) -> None:
    """
    Plot I/sigma histogram.

    Args:
        intensities: Array of intensity values
        sigmas: Array of sigma (error) values
        output_path: Output file path
    """
    # Calculate I/sigma, handling sigma=0 carefully
    with np.errstate(divide="ignore", invalid="ignore"):
        isigi = intensities / sigmas
        isigi = isigi[np.isfinite(isigi)]  # Remove inf and nan values

    if len(isigi) == 0:
        logger.warning("No valid I/sigma values found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(isigi, bins=50, alpha=0.7, color="orange", edgecolor="black")
    ax.set_xlabel("I/σ")
    ax.set_ylabel("Count")
    ax.set_title("I/σ Distribution")

    # Add statistics text
    mean_isigi = np.mean(isigi)
    median_isigi = np.median(isigi)
    ax.axvline(mean_isigi, color="red", linestyle="--", label=f"Mean: {mean_isigi:.2f}")
    ax.axvline(
        median_isigi, color="green", linestyle="--", label=f"Median: {median_isigi:.2f}"
    )
    ax.legend()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved I/sigma histogram to {output_path}")


def generate_intensity_correction_summary(
    npz_data: Dict[str, np.ndarray], output_path: Path
) -> None:
    """
    Generate simplified intensity correction effects summary.

    Since full step-by-step intensity transformation data is not typically
    available in the NPZ file, this creates a simplified summary.

    Args:
        npz_data: Loaded NPZ data dictionary
        output_path: Output file path for summary
    """
    intensities = npz_data["intensities"]
    sigmas = npz_data["sigmas"]
    q_vectors = npz_data["q_vectors"]

    # Select a random sample of points for detailed inspection
    n_sample = min(100, len(intensities))
    sample_indices = np.random.choice(len(intensities), n_sample, replace=False)

    with open(output_path, "w") as f:
        f.write("Intensity Correction Effects Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total diffuse points: {len(intensities)}\n")
        f.write(f"Sample size for detailed inspection: {n_sample}\n\n")

        f.write("Sample of corrected intensity values:\n")
        f.write("Index\tqx\tqy\tqz\tIntensity\tSigma\tI/σ\n")
        f.write("-" * 60 + "\n")

        for i, idx in enumerate(sample_indices[:20]):  # Show first 20
            qx, qy, qz = q_vectors[idx]
            intensity = intensities[idx]
            sigma = sigmas[idx]
            isigi = intensity / sigma if sigma > 0 else np.inf

            f.write(
                f"{idx}\t{qx:.4f}\t{qy:.4f}\t{qz:.4f}\t{intensity:.2f}\t{sigma:.2f}\t{isigi:.2f}\n"
            )

        if n_sample > 20:
            f.write(f"... and {n_sample - 20} more points\n")

        f.write("\n" + "=" * 40 + "\n")
        f.write("NOTE: Full step-by-step intensity transformation plotting\n")
        f.write("requires enhanced logging/output from DataExtractor.\n")

    logger.info(f"Saved intensity correction summary to {output_path}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        setup_logging_for_plots()

    logger.info("Starting diffuse extraction visual diagnostics")

    try:
        # Validate input files
        validate_input_files(args)

        # Create output directory
        output_dir = ensure_output_dir(args.output_dir)
        logger.info(f"Output directory: {output_dir}")

        # Load data
        logger.info("Loading input data...")
        npz_data = load_npz_data(args.npz_file)
        imageset = load_cbf_image(args.raw_image)
        experiment = load_experiment(args.expt)
        # Load mask files (currently only validating they exist)
        load_mask_pickle(args.total_mask)

        # Load optional data (for validation)
        if args.bragg_mask:
            load_mask_pickle(args.bragg_mask)

        if args.pixel_mask:
            load_mask_pickle(args.pixel_mask)

        if args.bg_map:
            load_background_map(args.bg_map)

        # Generate diagnostic plots
        logger.info("Generating diagnostic plots...")

        # Extract key data
        q_vectors = npz_data["q_vectors"]
        intensities = npz_data["intensities"]
        sigmas = npz_data["sigmas"]

        # Check if pixel coordinates are available
        has_pixel_coords = all(
            key in npz_data
            for key in [
                "original_panel_ids",
                "original_fast_coords",
                "original_slow_coords",
            ]
        )

        if has_pixel_coords:
            panel_ids = npz_data["original_panel_ids"]
            fast_coords = npz_data["original_fast_coords"]
            slow_coords = npz_data["original_slow_coords"]
            logger.info(
                "Pixel coordinates available - will generate full diagnostic suite"
            )
        else:
            logger.warning(
                "Pixel coordinates not available - skipping pixel overlay and heatmap plots"
            )

        # Load raw image data (use first panel)
        # Corrected logic:
        # ImageSetFactory.new() returns a list, so get the first ImageSet
        if isinstance(imageset, list):
            imageset_obj = imageset[0]
        else:
            imageset_obj = imageset

        raw_data_tuple_or_array = imageset_obj.get_raw_data(
            0
        )  # Get raw data for the 0-th image in the set
        if isinstance(raw_data_tuple_or_array, tuple):
            # Multi-panel detector, using data from the first panel for this plot
            raw_image_data = raw_data_tuple_or_array[0]
            logger.info(
                "Multi-panel detector: using data from panel 0 for diagnostic plot."
            )
        else:
            # Single-panel detector
            raw_image_data = raw_data_tuple_or_array

        # Get detector information for panel shapes
        detector = experiment.detector
        panel_0 = detector[0]  # First panel
        panel_shape = panel_0.get_image_size()[::-1]  # (height, width)

        # Plot 1: Diffuse pixel overlay (conditional on pixel coordinates)
        if has_pixel_coords:
            logger.info("Generating diffuse pixel overlay plot...")

            # Filter coordinates for panel 0
            panel_0_mask = panel_ids == 0
            if np.any(panel_0_mask):
                panel_0_coords = list(
                    zip(fast_coords[panel_0_mask], slow_coords[panel_0_mask])
                )

                # Apply subsampling for pixel overlay if necessary
                if len(panel_0_coords) > args.max_plot_points:
                    sample_indices = np.random.choice(
                        len(panel_0_coords), args.max_plot_points, replace=False
                    )
                    panel_0_coords_sampled = [panel_0_coords[i] for i in sample_indices]
                    overlay_title = f"Raw Image with Extracted Diffuse Pixels (Panel 0) - sampled {args.max_plot_points} of {len(panel_0_coords)} points"
                else:
                    panel_0_coords_sampled = panel_0_coords
                    overlay_title = "Raw Image with Extracted Diffuse Pixels (Panel 0)"

                plot_diffuse_pixel_overlay(
                    raw_image_data,
                    panel_id=0,
                    pixel_coords_for_panel=panel_0_coords_sampled,
                    title=overlay_title,
                    output_path=str(output_dir / "diffuse_pixel_overlay.png"),
                )
        else:
            logger.info("Skipping diffuse pixel overlay plot (no pixel coordinates)")

        # Plot 2: Intensity correction effects (simplified)
        logger.info("Generating intensity correction summary...")
        generate_intensity_correction_summary(
            npz_data, output_dir / "intensity_correction_summary.txt"
        )

        # Plot 3 & 4: Q-space coverage
        logger.info("Generating Q-space coverage plots...")
        plot_q_projections(q_vectors, intensities, output_dir, args.max_plot_points)
        plot_radial_q(
            q_vectors,
            intensities,
            output_dir / "radial_q_distribution.png",
            args.max_plot_points,
        )

        # Plot 5: Intensity distribution
        logger.info("Generating intensity distribution histogram...")
        plot_intensity_histogram(intensities, output_dir / "intensity_histogram.png")

        # Plot 6: Intensity heatmap (conditional on pixel coordinates)
        if has_pixel_coords:
            logger.info("Generating intensity heatmap...")

            # Generate heatmap for panel 0
            panel_0_mask = panel_ids == 0
            if np.any(panel_0_mask):
                panel_0_coords = list(
                    zip(fast_coords[panel_0_mask], slow_coords[panel_0_mask])
                )
                panel_0_intensities = intensities[panel_0_mask]

                plot_intensity_heatmap(
                    panel_id=0,
                    pixel_coords_for_panel=panel_0_coords,
                    intensities_for_panel=panel_0_intensities,
                    panel_shape=panel_shape,
                    title="Intensity Heatmap (Panel 0)",
                    output_path=output_dir / "intensity_heatmap_panel_0.png",
                    max_points=args.max_plot_points,
                )
        else:
            logger.info("Skipping intensity heatmap plot (no pixel coordinates)")

        # Plot 7 & 8: Sigma analysis
        logger.info("Generating sigma analysis plots...")
        plot_sigma_vs_intensity(
            intensities,
            sigmas,
            output_dir / "sigma_vs_intensity.png",
            args.max_plot_points,
        )
        plot_isigi_histogram(intensities, sigmas, output_dir / "isigi_histogram.png")

        # Generate summary report
        logger.info("Generating summary report...")
        summary_path = output_dir / "extraction_diagnostics_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Diffuse Extraction Visual Diagnostics Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write("Input files:\n")
            f.write(f"  Raw image: {args.raw_image}\n")
            f.write(f"  Experiment: {args.expt}\n")
            f.write(f"  Total mask: {args.total_mask}\n")
            f.write(f"  NPZ file: {args.npz_file}\n\n")

            f.write("Data summary:\n")
            f.write(f"  Total diffuse points: {len(intensities)}\n")
            f.write(
                f"  Q-vector range: [{q_vectors.min():.4f}, {q_vectors.max():.4f}] Å⁻¹\n"
            )
            f.write(
                f"  Intensity range: [{intensities.min():.2f}, {intensities.max():.2f}]\n"
            )
            f.write(f"  Sigma range: [{sigmas.min():.2f}, {sigmas.max():.2f}]\n")
            f.write(
                f"  Mean I/σ: {np.mean(intensities/np.where(sigmas > 0, sigmas, np.inf)):.2f}\n"
            )
            f.write(f"  Pixel coordinates available: {has_pixel_coords}\n\n")

            f.write("Generated plots:\n")
            if has_pixel_coords:
                f.write("  ✓ diffuse_pixel_overlay.png\n")
                f.write("  ✓ intensity_heatmap_panel_0.png\n")
            else:
                f.write("  ✗ diffuse_pixel_overlay.png (no pixel coordinates)\n")
                f.write("  ✗ intensity_heatmap_panel_0.png (no pixel coordinates)\n")

            f.write("  ✓ intensity_correction_summary.txt\n")
            f.write("  ✓ q_projection_qx_qy.png\n")
            f.write("  ✓ q_projection_qx_qz.png\n")
            f.write("  ✓ q_projection_qy_qz.png\n")
            f.write("  ✓ radial_q_distribution.png\n")
            f.write("  ✓ intensity_histogram.png\n")
            f.write("  ✓ sigma_vs_intensity.png\n")
            f.write("  ✓ isigi_histogram.png\n")

        logger.info("Diffuse extraction visual diagnostics completed successfully")
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        # Clean up
        close_all_figures()


if __name__ == "__main__":
    main()
