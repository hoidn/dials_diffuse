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
matplotlib.use('Agg')
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
    cmap: str = 'viridis'
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
    if hasattr(image_data, 'as_numpy_array'):
        img_array = image_data.as_numpy_array()
    else:
        img_array = np.array(image_data)
    
    # Ensure 2D array
    if len(img_array.shape) == 1:
        # Try to infer shape from flex array
        if hasattr(image_data, 'accessor'):
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
        origin='lower',  # Use lower origin to match DIALS convention
        aspect='equal'
    )
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Intensity' if not log_scale else 'Intensity (log)')
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Fast axis (pixels)')
    ax.set_ylabel('Slow axis (pixels)')
    
    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    
    return fig


def plot_mask_overlay(
    image_data: Union[Any, np.ndarray],
    mask_data: Union[Any, np.ndarray],
    title: str = "Image with Mask Overlay",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    mask_color: str = 'red',
    mask_alpha: float = 0.3,
    log_scale: bool = False
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
    if hasattr(image_data, 'as_numpy_array'):
        img_array = image_data.as_numpy_array()
    else:
        img_array = np.array(image_data)
    
    if hasattr(mask_data, 'as_numpy_array'):
        mask_array = mask_data.as_numpy_array()
    else:
        mask_array = np.array(mask_data)
    
    # Handle 1D arrays
    if len(img_array.shape) == 1 and hasattr(image_data, 'accessor'):
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
    
    im = ax.imshow(
        img_array,
        cmap='gray',
        norm=norm,
        origin='lower',
        aspect='equal'
    )
    
    # Overlay mask
    masked_regions = np.ma.masked_where(~mask_array, mask_array)
    ax.imshow(
        masked_regions,
        cmap=plt.cm.get_cmap(mask_color).with_extremes(under='none'),
        alpha=mask_alpha,
        origin='lower',
        aspect='equal'
    )
    
    # Add colorbar for image
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Fast axis (pixels)')
    ax.set_ylabel('Slow axis (pixels)')
    
    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    
    return fig


def plot_spot_overlay(
    image_data: Union[Any, np.ndarray],
    spot_positions: List[Tuple[float, float]],
    title: str = "Image with Spot Overlay",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    spot_color: str = 'red',
    spot_size: float = 20,
    log_scale: bool = False,
    predicted_positions: Optional[List[Tuple[float, float]]] = None,
    predicted_color: str = 'blue'
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
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy array
    if hasattr(image_data, 'as_numpy_array'):
        img_array = image_data.as_numpy_array()
    else:
        img_array = np.array(image_data)
    
    # Handle 1D arrays
    if len(img_array.shape) == 1 and hasattr(image_data, 'accessor'):
        accessor = image_data.accessor()
        height, width = accessor.all()
        img_array = img_array.reshape(height, width)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot background image
    norm = None
    if log_scale:
        img_array = np.where(img_array <= 0, 1e-6, img_array)
        norm = LogNorm(vmin=img_array.min(), vmax=img_array.max())
    
    im = ax.imshow(
        img_array,
        cmap='gray',
        norm=norm,
        origin='lower',
        aspect='equal'
    )
    
    # Plot observed spots
    if spot_positions:
        x_coords, y_coords = zip(*spot_positions)
        ax.scatter(
            x_coords, y_coords,
            c=spot_color,
            s=spot_size,
            marker='o',
            alpha=0.7,
            label='Observed spots',
            edgecolors='white',
            linewidth=0.5
        )
    
    # Plot predicted spots if provided
    if predicted_positions:
        pred_x, pred_y = zip(*predicted_positions)
        ax.scatter(
            pred_x, pred_y,
            c=predicted_color,
            s=spot_size,
            marker='x',
            alpha=0.7,
            label='Predicted spots',
            linewidth=1.5
        )
    
    # Add colorbar for image
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # Add legend if we have spots
    if spot_positions or predicted_positions:
        ax.legend()
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Fast axis (pixels)')
    ax.set_ylabel('Slow axis (pixels)')
    
    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    
    return fig


def plot_multi_panel_comparison(
    images: List[Union[Any, np.ndarray]],
    titles: List[str],
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 5),
    log_scale: bool = False,
    cmap: str = 'viridis'
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
        if hasattr(image_data, 'as_numpy_array'):
            img_array = image_data.as_numpy_array()
        else:
            img_array = np.array(image_data)
        
        # Handle 1D arrays
        if len(img_array.shape) == 1 and hasattr(image_data, 'accessor'):
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
            img_array,
            cmap=cmap,
            norm=norm,
            origin='lower',
            aspect='equal'
        )
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
        
        # Set title and labels
        axes[i].set_title(title)
        axes[i].set_xlabel('Fast axis (pixels)')
        axes[i].set_ylabel('Slow axis (pixels)')
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.close('all')


def setup_logging_for_plots():
    """Set up logging configuration for plotting scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )