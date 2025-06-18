"""
Plotting utilities for visual diagnostics.

This module provides reusable plotting functions for visualizing detector images,
masks, and other crystallographic data with matplotlib.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Any

# Set matplotlib to non-interactive backend for automation
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

logger = logging.getLogger(__name__)


def plot_detector_image(
    image_data: Union[Any, np.ndarray],
    title: str = "Detector Image",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    log_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Plot a single detector panel image.

    Args:
        image_data: Image data as numpy array or DIALS flex array
        title: Title for the plot
        output_path: Path to save the plot (optional)
        figsize: Figure size (width, height) in inches
        log_scale: Whether to use logarithmic color scale
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        cmap: Colormap name

    Returns:
        matplotlib Figure object
    """
    # Convert DIALS flex arrays to numpy if needed
    if hasattr(image_data, "as_numpy_array"):
        img_array = image_data.as_numpy_array()
    else:
        img_array = np.array(image_data)

    # Ensure 2D array
    if len(img_array.shape) == 1:
        # Try to infer shape from flex array
        if hasattr(image_data, "accessor"):
            accessor = image_data.accessor()
            height, width = accessor.all()
            img_array = img_array.reshape(height, width)
        else:
            raise ValueError("Cannot determine image dimensions")

    fig, ax = plt.subplots(figsize=figsize)

    # Set up color scaling
    norm = None
    if log_scale:
        # Avoid log(0) by setting minimum to small positive value
        img_array = np.where(img_array <= 0, 1e-6, img_array)
        norm = LogNorm(vmin=vmin or img_array.min(), vmax=vmax or img_array.max())

    # Create the image plot
    im = ax.imshow(
        img_array,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        origin="lower",  # Use lower origin to match DIALS convention
        aspect="equal",
    )

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Intensity" if not log_scale else "Intensity (log)")

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Fast axis (pixels)")
    ax.set_ylabel("Slow axis (pixels)")

    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")

    return fig


def plot_mask_overlay(
    image_data: Union[Any, np.ndarray],
    mask_data: Union[Any, np.ndarray],
    title: str = "Image with Mask Overlay",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    mask_color: str = "red",
    mask_alpha: float = 0.3,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot detector image with mask overlay.

    Args:
        image_data: Background image data
        mask_data: Boolean mask data (True = masked regions)
        title: Title for the plot
        output_path: Path to save the plot (optional)
        figsize: Figure size (width, height) in inches
        mask_color: Color for mask overlay
        mask_alpha: Transparency of mask overlay (0-1)
        log_scale: Whether to use logarithmic color scale for image

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy arrays
    if hasattr(image_data, "as_numpy_array"):
        img_array = image_data.as_numpy_array()
    else:
        img_array = np.array(image_data)

    if hasattr(mask_data, "as_numpy_array"):
        mask_array = mask_data.as_numpy_array()
    else:
        mask_array = np.array(mask_data)

    # Handle 1D arrays
    if len(img_array.shape) == 1 and hasattr(image_data, "accessor"):
        accessor = image_data.accessor()
        height, width = accessor.all()
        img_array = img_array.reshape(height, width)
        mask_array = mask_array.reshape(height, width)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot background image
    norm = None
    if log_scale:
        img_array = np.where(img_array <= 0, 1e-6, img_array)
        norm = LogNorm(vmin=img_array.min(), vmax=img_array.max())

    im = ax.imshow(img_array, cmap="gray", norm=norm, origin="lower", aspect="equal")

    # Overlay mask
    masked_regions = np.ma.masked_where(~mask_array, mask_array)
    ax.imshow(
        masked_regions,
        cmap=plt.cm.get_cmap(mask_color).with_extremes(under="none"),
        alpha=mask_alpha,
        origin="lower",
        aspect="equal",
    )

    # Add colorbar for image
    plt.colorbar(im, ax=ax, label="Intensity")

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Fast axis (pixels)")
    ax.set_ylabel("Slow axis (pixels)")

    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")

    return fig


def plot_spot_overlay(
    image_data: Union[Any, np.ndarray],
    spot_positions: List[Tuple[float, float]],
    title: str = "Image with Spot Overlay",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    spot_color: str = "red",
    spot_size: float = 20,
    log_scale: bool = False,
    predicted_positions: Optional[List[Tuple[float, float]]] = None,
    predicted_color: str = "blue",
    max_points: Optional[int] = None,
) -> plt.Figure:
    """
    Plot detector image with spot position overlays.

    Args:
        image_data: Background image data
        spot_positions: List of (x, y) pixel coordinates for observed spots
        title: Title for the plot
        output_path: Path to save the plot (optional)
        figsize: Figure size (width, height) in inches
        spot_color: Color for observed spot markers
        spot_size: Size of spot markers
        log_scale: Whether to use logarithmic color scale
        predicted_positions: Optional list of predicted spot positions
        predicted_color: Color for predicted spot markers
        max_points: Maximum number of points to plot for performance

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy array
    if hasattr(image_data, "as_numpy_array"):
        img_array = image_data.as_numpy_array()
    else:
        img_array = np.array(image_data)

    # Handle 1D arrays
    if len(img_array.shape) == 1 and hasattr(image_data, "accessor"):
        accessor = image_data.accessor()
        height, width = accessor.all()
        img_array = img_array.reshape(height, width)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot background image
    norm = None
    if log_scale:
        img_array = np.where(img_array <= 0, 1e-6, img_array)
        norm = LogNorm(vmin=img_array.min(), vmax=img_array.max())

    im = ax.imshow(img_array, cmap="gray", norm=norm, origin="lower", aspect="equal")

    # Apply subsampling to spot positions if necessary
    sampled_spots = spot_positions
    sampled_predicted = predicted_positions
    sampled_note = ""

    if max_points and spot_positions and len(spot_positions) > max_points:
        indices = np.random.choice(len(spot_positions), max_points, replace=False)
        sampled_spots = [spot_positions[i] for i in indices]
        sampled_note = f" (sampled {max_points} of {len(spot_positions)} spots)"

    if max_points and predicted_positions and len(predicted_positions) > max_points:
        pred_indices = np.random.choice(
            len(predicted_positions), max_points, replace=False
        )
        sampled_predicted = [predicted_positions[i] for i in pred_indices]
        if not sampled_note:  # Only add note if not already added for observed spots
            sampled_note = (
                f" (sampled {max_points} of {len(predicted_positions)} predicted spots)"
            )

    # Plot observed spots
    if sampled_spots:
        x_coords, y_coords = zip(*sampled_spots)
        ax.scatter(
            x_coords,
            y_coords,
            c=spot_color,
            s=spot_size,
            marker="o",
            alpha=0.7,
            label="Observed spots",
            edgecolors="white",
            linewidth=0.5,
        )

    # Plot predicted spots if provided
    if sampled_predicted:
        pred_x, pred_y = zip(*sampled_predicted)
        ax.scatter(
            pred_x,
            pred_y,
            c=predicted_color,
            s=spot_size,
            marker="x",
            alpha=0.7,
            label="Predicted spots",
            linewidth=1.5,
        )

    # Add colorbar for image
    plt.colorbar(im, ax=ax, label="Intensity")

    # Add legend if we have spots
    if spot_positions or predicted_positions:
        ax.legend()

    # Set title and labels
    ax.set_title(title + sampled_note)
    ax.set_xlabel("Fast axis (pixels)")
    ax.set_ylabel("Slow axis (pixels)")

    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")

    return fig


def plot_multi_panel_comparison(
    images: List[Union[Any, np.ndarray]],
    titles: List[str],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 5),
    log_scale: bool = False,
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Plot multiple images side by side for comparison.

    Args:
        images: List of image arrays to plot
        titles: List of titles for each subplot
        output_path: Path to save the plot (optional)
        figsize: Figure size (width, height) in inches
        log_scale: Whether to use logarithmic color scale
        cmap: Colormap name

    Returns:
        matplotlib Figure object
    """
    n_images = len(images)
    if n_images != len(titles):
        raise ValueError("Number of images must match number of titles")

    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]

    for i, (image_data, title) in enumerate(zip(images, titles)):
        # Convert to numpy array
        if hasattr(image_data, "as_numpy_array"):
            img_array = image_data.as_numpy_array()
        else:
            img_array = np.array(image_data)

        # Handle 1D arrays
        if len(img_array.shape) == 1 and hasattr(image_data, "accessor"):
            accessor = image_data.accessor()
            height, width = accessor.all()
            img_array = img_array.reshape(height, width)

        # Set up color scaling
        norm = None
        if log_scale:
            img_array = np.where(img_array <= 0, 1e-6, img_array)
            norm = LogNorm(vmin=img_array.min(), vmax=img_array.max())

        # Plot image
        im = axes[i].imshow(
            img_array, cmap=cmap, norm=norm, origin="lower", aspect="equal"
        )

        # Add colorbar
        plt.colorbar(im, ax=axes[i])

        # Set title and labels
        axes[i].set_title(title)
        axes[i].set_xlabel("Fast axis (pixels)")
        axes[i].set_ylabel("Slow axis (pixels)")

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")

    return fig


def ensure_output_dir(output_dir: str) -> Path:
    """
    Ensure output directory exists and return Path object.

    Args:
        output_dir: Output directory path

    Returns:
        Path object for the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def close_all_figures():
    """Close all matplotlib figures to free memory."""
    plt.close("all")


def plot_3d_grid_slice(
    grid_data_3d: np.ndarray,
    slice_dim_idx: int,
    slice_val_idx: int,
    title: str,
    output_path: str,
    cmap: str = "viridis",
    norm: Optional[Any] = None,
    xlabel: str = "H",
    ylabel: str = "K",
    aspect: str = "auto",
) -> plt.Figure:
    """
    Plot a 2D slice through a 3D grid.

    Args:
        grid_data_3d: 3D numpy array with shape (H, K, L)
        slice_dim_idx: Dimension to slice (0=H, 1=K, 2=L)
        slice_val_idx: Index along slice dimension
        title: Plot title
        output_path: Path to save the plot
        cmap: Colormap name
        norm: Color normalization (e.g., LogNorm)
        xlabel: X-axis label
        ylabel: Y-axis label
        aspect: Aspect ratio ('auto', 'equal')

    Returns:
        matplotlib Figure object
    """
    # Extract the 2D slice
    if slice_dim_idx == 0:  # H slice
        slice_data = grid_data_3d[slice_val_idx, :, :]
        xlabel = "K"
        ylabel = "L"
    elif slice_dim_idx == 1:  # K slice
        slice_data = grid_data_3d[:, slice_val_idx, :]
        xlabel = "H"
        ylabel = "L"
    elif slice_dim_idx == 2:  # L slice
        slice_data = grid_data_3d[:, :, slice_val_idx]
        xlabel = "H"
        ylabel = "K"
    else:
        raise ValueError("slice_dim_idx must be 0, 1, or 2")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the image plot
    im = ax.imshow(
        slice_data,
        cmap=cmap,
        norm=norm,
        origin="lower",
        aspect=aspect,
        interpolation="nearest",
    )

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved 3D grid slice plot to {output_path}")

    return fig


def plot_radial_average(
    q_magnitudes: np.ndarray,
    intensities: np.ndarray,
    num_bins: int,
    title: str,
    output_path: str,
    sigmas: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Plot radial average of intensities vs q-magnitude.

    Args:
        q_magnitudes: Array of q-magnitude values
        intensities: Array of intensity values
        num_bins: Number of bins for radial averaging
        title: Plot title
        output_path: Path to save the plot
        sigmas: Optional array of sigma values for error bars

    Returns:
        matplotlib Figure object
    """
    # Remove invalid data
    valid_mask = np.isfinite(q_magnitudes) & np.isfinite(intensities)
    q_valid = q_magnitudes[valid_mask]
    I_valid = intensities[valid_mask]

    if sigmas is not None:
        sigmas_valid = sigmas[valid_mask]
    else:
        sigmas_valid = None

    if len(q_valid) == 0:
        logger.warning("No valid data for radial average plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.set_title(title)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved radial average plot (no data) to {output_path}")
        return fig

    # Create bins
    q_min, q_max = np.min(q_valid), np.max(q_valid)
    if q_min == q_max:
        q_max = q_min + 1e-6  # Avoid zero width

    bin_edges = np.linspace(q_min, q_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Bin the data
    bin_indices = np.digitize(q_valid, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    mean_intensities = []
    std_intensities = []

    for i in range(num_bins):
        mask = bin_indices == i
        if np.any(mask):
            I_bin = I_valid[mask]
            mean_intensities.append(np.mean(I_bin))

            if sigmas_valid is not None:
                # Propagate uncertainties: std error of mean
                weights = 1.0 / (sigmas_valid[mask] ** 2)
                weighted_mean = np.average(I_bin, weights=weights)
                weighted_std = np.sqrt(1.0 / np.sum(weights))
                std_intensities.append(weighted_std)
            else:
                std_intensities.append(np.std(I_bin) / np.sqrt(len(I_bin)))
        else:
            mean_intensities.append(np.nan)
            std_intensities.append(np.nan)

    mean_intensities = np.array(mean_intensities)
    std_intensities = np.array(std_intensities)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot with error bars if available
    if sigmas is not None:
        ax.errorbar(
            bin_centers,
            mean_intensities,
            yerr=std_intensities,
            fmt="o-",
            markersize=4,
            linewidth=1,
            capsize=3,
        )
    else:
        ax.plot(bin_centers, mean_intensities, "o-", markersize=4, linewidth=1)

    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel("Mean Intensity")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved radial average plot to {output_path}")

    return fig


def plot_parameter_vs_index(
    param_values: np.ndarray,
    index_values: np.ndarray,
    title: str,
    param_label: str,
    index_label: str,
    output_path: str,
) -> plt.Figure:
    """
    Plot parameter values vs index (e.g., scaling factors vs still index).

    Args:
        param_values: Array of parameter values
        index_values: Array of index values
        title: Plot title
        param_label: Y-axis label for parameter
        index_label: X-axis label for index
        output_path: Path to save the plot

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(index_values, param_values, "o-", markersize=4, linewidth=1)

    ax.set_xlabel(index_label)
    ax.set_ylabel(param_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_val = np.mean(param_values)
    std_val = np.std(param_values)
    ax.axhline(
        mean_val, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_val:.3f}"
    )
    ax.axhline(
        mean_val + std_val,
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"±1σ: {std_val:.3f}",
    )
    ax.axhline(mean_val - std_val, color="orange", linestyle=":", alpha=0.7)
    ax.legend()

    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved parameter vs index plot to {output_path}")

    return fig


def plot_smoother_curve(
    smoother_eval_func: callable,
    x_range: Tuple[float, float],
    num_points: int,
    title: str,
    output_path: str,
    control_points_x: Optional[np.ndarray] = None,
    control_points_y: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Plot a smoother curve with optional control points.

    Args:
        smoother_eval_func: Function that evaluates the smoother at given x values
        x_range: Tuple of (x_min, x_max) for plotting range
        num_points: Number of points to evaluate the curve
        title: Plot title
        output_path: Path to save the plot
        control_points_x: Optional x-coordinates of control points
        control_points_y: Optional y-coordinates of control points

    Returns:
        matplotlib Figure object
    """
    # Generate evaluation points
    x_eval = np.linspace(x_range[0], x_range[1], num_points)

    try:
        y_eval = smoother_eval_func(x_eval)
    except Exception as e:
        logger.error(f"Failed to evaluate smoother function: {e}")
        # Create a fallback plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Failed to evaluate smoother: {e}", ha="center", va="center")
        ax.set_title(title)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved smoother curve plot (error) to {output_path}")
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the smooth curve
    ax.plot(x_eval, y_eval, "-", linewidth=2, label="Smoother curve")

    # Plot control points if provided
    if control_points_x is not None and control_points_y is not None:
        ax.scatter(
            control_points_x,
            control_points_y,
            c="red",
            s=50,
            marker="o",
            label="Control points",
            zorder=5,
        )

    ax.set_xlabel("Q (Å⁻¹)")
    ax.set_ylabel("Correction Factor")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved smoother curve plot to {output_path}")

    return fig


def setup_logging_for_plots():
    """Set up logging configuration for plotting scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
