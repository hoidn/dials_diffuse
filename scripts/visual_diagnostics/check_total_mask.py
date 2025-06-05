#!/usr/bin/env python3
"""
Visual diagnostic script for total mask generation (Module 1.S.3).

This script generates Bragg masks and combines them with pixel masks to create
total masks for diffuse scattering analysis. It provides visualizations to verify
that Bragg peak regions are properly masked and diffuse regions are preserved.
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from plot_utils import (
    plot_detector_image,
    plot_mask_overlay,
    plot_multi_panel_comparison,
    ensure_output_dir,
    close_all_figures,
    setup_logging_for_plots
)

# Import diffusepipe components
from diffusepipe.masking.bragg_mask_generator import (
    BraggMaskGenerator,
    create_default_bragg_mask_config,
    validate_mask_compatibility
)
from diffusepipe.masking.pixel_mask_generator import (
    PixelMaskGenerator,
    create_default_static_params,
    create_default_dynamic_params
)

logger = logging.getLogger(__name__)


def load_dials_data(image_path: str, expt_path: str, refl_path: str):
    """
    Load raw image and DIALS processing outputs.
    
    Args:
        image_path: Path to raw image file
        expt_path: Path to .expt file
        refl_path: Path to .refl file
        
    Returns:
        Tuple of (image_set, experiments, reflections) or (None, None, None) if failed
    """
    try:
        from dxtbx.imageset import ImageSetFactory
        from dxtbx.model.experiment_list import ExperimentListFactory
        from dials.array_family import flex
        
        # Load raw image
        image_set = ImageSetFactory.new([image_path])[0]
        
        # Load experiments
        experiments = ExperimentListFactory.from_json_file(expt_path)
        
        # Load reflections
        reflections = flex.reflection_table.from_file(refl_path)
        
        logger.info(f"Loaded image, {len(experiments)} experiments, and {len(reflections)} reflections")
        return image_set, experiments, reflections
        
    except ImportError as e:
        logger.error(f"Failed to import DIALS components: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"Failed to load DIALS data: {e}")
        return None, None, None


def parse_bragg_config(config_str: str) -> Dict[str, Any]:
    """
    Parse Bragg mask configuration from JSON string or file.
    
    Args:
        config_str: JSON string or path to JSON file
        
    Returns:
        Configuration dictionary
    """
    try:
        # Try to parse as JSON string first
        if config_str and config_str.strip().startswith('{'):
            return json.loads(config_str)
        
        # Otherwise treat as file path
        if config_str:
            config_path = Path(config_str)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
        
        # Return default if no config provided
        return create_default_bragg_mask_config()
            
    except Exception as e:
        logger.error(f"Failed to parse Bragg mask config: {e}")
        return create_default_bragg_mask_config()


def generate_pixel_mask(detector, representative_images: list):
    """
    Generate global pixel mask using default parameters.
    
    Args:
        detector: DIALS Detector object
        representative_images: List of ImageSet objects
        
    Returns:
        Combined pixel mask or None if failed
    """
    try:
        generator = PixelMaskGenerator()
        static_params = create_default_static_params()
        dynamic_params = create_default_dynamic_params()
        
        pixel_mask = generator.generate_combined_pixel_mask(
            detector, static_params, representative_images, dynamic_params
        )
        
        logger.info("Generated global pixel mask")
        return pixel_mask
        
    except Exception as e:
        logger.error(f"Failed to generate pixel mask: {e}")
        return None


def create_total_mask_visualizations(
    image_set,
    experiments,
    reflections,
    pixel_mask,
    bragg_config: Dict[str, Any],
    output_dir: str,
    use_option_b: bool = False
) -> bool:
    """
    Create visualizations for total mask generation.
    
    Args:
        image_set: Raw image data
        experiments: DIALS experiments
        reflections: DIALS reflections
        pixel_mask: Global pixel mask
        bragg_config: Bragg mask generation configuration
        output_dir: Directory to save plots
        use_option_b: Whether to use shoebox-based Bragg masking (Option B)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = ensure_output_dir(output_dir)
        generator = BraggMaskGenerator()
        
        # Get raw image data (first image, first panel)
        raw_data = image_set.get_raw_data(0)
        if len(raw_data) > 1:
            logger.info(f"Multi-panel detector detected ({len(raw_data)} panels), using first panel")
        image_data = raw_data[0]
        
        # Generate Bragg mask
        experiment = experiments[0]
        
        if use_option_b:
            logger.info("Generating Bragg mask using shoebox data (Option B)...")
            bragg_mask = generator.generate_bragg_mask_from_shoeboxes(
                reflections, experiment.detector
            )
        else:
            logger.info("Generating Bragg mask using dials.generate_mask (Option A)...")
            bragg_mask = generator.generate_bragg_mask_from_spots(
                experiment, reflections, bragg_config
            )
        
        # Generate total mask
        logger.info("Generating total mask...")
        total_mask = generator.get_total_mask_for_still(bragg_mask, pixel_mask)
        
        # Validate mask compatibility
        if not validate_mask_compatibility(bragg_mask, pixel_mask):
            logger.error("Bragg and pixel masks are incompatible")
            return False
        
        # Create visualizations for first panel
        panel_idx = 0
        bragg_panel = bragg_mask[panel_idx]
        pixel_panel = pixel_mask[panel_idx]
        total_panel = total_mask[panel_idx]
        
        # Plot 1: Raw image alone
        logger.info("Creating raw image plot...")
        plot_detector_image(
            image_data,
            title="Raw Still Image",
            output_path=str(output_path / "raw_image.png"),
            log_scale=True
        )
        
        # Plot 2: Raw image with Bragg mask overlay
        logger.info("Creating Bragg mask overlay...")
        plot_mask_overlay(
            image_data,
            bragg_panel,
            title="Raw Image with Bragg Mask (red = masked Bragg regions)",
            output_path=str(output_path / "image_with_bragg_mask.png"),
            mask_color='red',
            mask_alpha=0.4,
            log_scale=True
        )
        
        # Plot 3: Raw image with total mask overlay (diffuse regions)
        logger.info("Creating total mask overlay...")
        plot_mask_overlay(
            image_data,
            total_panel,
            title="Raw Image with Total Mask (blue = good diffuse pixels)",
            output_path=str(output_path / "image_with_total_diffuse_mask.png"),
            mask_color='blue',
            mask_alpha=0.3,
            log_scale=True
        )
        
        # Plot 4: Masked image showing only diffuse pixels
        logger.info("Creating diffuse-only image...")
        try:
            # Convert masks to numpy arrays for multiplication
            import numpy as np
            
            if hasattr(image_data, 'as_numpy_array'):
                img_array = image_data.as_numpy_array()
            else:
                img_array = np.array(image_data)
            
            if hasattr(total_panel, 'as_numpy_array'):
                mask_array = total_panel.as_numpy_array()
            else:
                mask_array = np.array(total_panel)
            
            # Handle 1D arrays
            if len(img_array.shape) == 1 and hasattr(image_data, 'accessor'):
                accessor = image_data.accessor()
                height, width = accessor.all()
                img_array = img_array.reshape(height, width)
                mask_array = mask_array.reshape(height, width)
            
            # Create masked image
            masked_image = img_array * mask_array.astype(float)
            
            plot_detector_image(
                masked_image,
                title="Diffuse Pixels Only (after total masking)",
                output_path=str(output_path / "image_diffuse_pixels_only.png"),
                log_scale=True
            )
            
        except Exception as e:
            logger.warning(f"Failed to create diffuse-only image: {e}")
        
        # Plot 5: Individual masks comparison
        logger.info("Creating mask comparison...")
        plot_multi_panel_comparison(
            [pixel_panel, bragg_panel, total_panel],
            ["Global Pixel Mask", "Bragg Mask", "Total Mask (diffuse regions)"],
            output_path=str(output_path / "mask_comparison.png"),
            figsize=(18, 6),
            cmap='RdYlBu'
        )
        
        # Create individual mask plots
        plot_detector_image(
            pixel_panel,
            title="Global Pixel Mask",
            output_path=str(output_path / "global_pixel_mask.png"),
            cmap='RdYlBu'
        )
        
        plot_detector_image(
            bragg_panel,
            title="Bragg Mask",
            output_path=str(output_path / "bragg_mask.png"),
            cmap='RdYlBu'
        )
        
        plot_detector_image(
            total_panel,
            title="Total Mask (Diffuse Regions)",
            output_path=str(output_path / "total_diffuse_mask.png"),
            cmap='RdYlBu'
        )
        
        # Generate summary information
        info_file = output_path / "total_mask_info.txt"
        with open(info_file, 'w') as f:
            f.write("Total Mask Generation Visual Check Results\n")
            f.write("=" * 43 + "\n\n")
            
            # Get panel size
            panel_size = experiment.detector[panel_idx].get_image_size()
            total_pixels = panel_size[0] * panel_size[1]
            
            # Count pixels in each mask
            pixel_good = pixel_panel.count(True)
            bragg_masked = bragg_panel.count(True)
            total_good = total_panel.count(True)
            
            f.write(f"Panel {panel_idx} ({panel_size[0]}x{panel_size[1]} pixels):\n")
            f.write(f"  Total pixels: {total_pixels}\n")
            f.write(f"  Global pixel mask good: {pixel_good} ({pixel_good/total_pixels*100:.1f}%)\n")
            f.write(f"  Bragg mask flagged: {bragg_masked} ({bragg_masked/total_pixels*100:.1f}%)\n")
            f.write(f"  Total mask good (diffuse): {total_good} ({total_good/total_pixels*100:.1f}%)\n")
            f.write(f"  Pixels excluded: {total_pixels - total_good} ({(total_pixels-total_good)/total_pixels*100:.1f}%)\n\n")
            
            # Reflection statistics
            if reflections:
                f.write(f"Reflection statistics:\n")
                f.write(f"  Total reflections: {len(reflections)}\n")
                
                if reflections.has_key('partiality'):
                    partialities = reflections['partiality']
                    f.write(f"  Partiality mean: {partialities.mean():.3f}\n")
                    f.write(f"  Partiality range: {partialities.min():.3f} - {partialities.max():.3f}\n")
                
                if reflections.has_key('intensity.sum.value'):
                    intensities = reflections['intensity.sum.value']
                    f.write(f"  Intensity mean: {intensities.mean():.1f}\n")
                    f.write(f"  Intensity range: {intensities.min():.1f} - {intensities.max():.1f}\n")
            
            f.write(f"\nBragg mask generation method: {'Option B (shoeboxes)' if use_option_b else 'Option A (dials.generate_mask)'}\n")
            f.write(f"Bragg mask configuration: {bragg_config}\n")
        
        logger.info(f"Total mask info saved to {info_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create total mask visualizations: {e}")
        return False


def main():
    """Main function for total mask visual checks."""
    parser = argparse.ArgumentParser(
        description="Visual diagnostic for total mask generation (Bragg + pixel masks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic total mask check
  python check_total_mask.py --raw-image image.cbf --expt processed.expt --refl processed.refl --output-dir total_mask_check
  
  # Using shoebox-based Bragg masking (Option B)
  python check_total_mask.py --raw-image image.cbf --expt processed.expt --refl processed.refl \\
    --use-option-b --output-dir total_mask_check
  
  # With custom Bragg mask configuration
  python check_total_mask.py --raw-image image.cbf --expt processed.expt --refl processed.refl \\
    --bragg-config '{"border": 3, "algorithm": "simple"}' --output-dir total_mask_check
  
  # Using existing test data
  python check_total_mask.py \\
    --raw-image ../../747/lys_nitr_10_6_0491.cbf \\
    --expt ../../lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt \\
    --refl ../../lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.refl \\
    --output-dir total_mask_visual_check
        """
    )
    
    parser.add_argument(
        '--raw-image',
        required=True,
        help='Path to raw still image file (e.g., CBF)'
    )
    
    parser.add_argument(
        '--expt',
        required=True,
        help='Path to DIALS experiment file (.expt)'
    )
    
    parser.add_argument(
        '--refl',
        required=True,
        help='Path to DIALS reflection file (.refl)'
    )
    
    parser.add_argument(
        '--bragg-config',
        help='Bragg mask configuration (JSON string or file path)'
    )
    
    parser.add_argument(
        '--use-option-b',
        action='store_true',
        help='Use shoebox-based Bragg masking (Option B) instead of dials.generate_mask (Option A)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='total_mask_visual_check',
        help='Output directory for plots (default: total_mask_visual_check)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging_for_plots()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting total mask visual check...")
    
    # Validate input files
    for file_path in [args.raw_image, args.expt, args.refl]:
        if not Path(file_path).exists():
            logger.error(f"Input file not found: {file_path}")
            return 1
    
    try:
        # Load DIALS data
        logger.info(f"Loading DIALS data...")
        image_set, experiments, reflections = load_dials_data(
            args.raw_image, args.expt, args.refl
        )
        if image_set is None or experiments is None or reflections is None:
            logger.error("Failed to load DIALS data")
            return 1
        
        # Generate global pixel mask
        logger.info("Generating global pixel mask...")
        pixel_mask = generate_pixel_mask(experiments[0].detector, [image_set])
        if pixel_mask is None:
            logger.error("Failed to generate pixel mask")
            return 1
        
        # Parse Bragg mask configuration
        bragg_config = parse_bragg_config(args.bragg_config)
        logger.info(f"Bragg mask config: {bragg_config}")
        
        # Create visualizations
        logger.info(f"Creating visualizations in: {args.output_dir}")
        success = create_total_mask_visualizations(
            image_set, experiments, reflections, pixel_mask,
            bragg_config, args.output_dir, args.use_option_b
        )
        
        if success:
            logger.info(f"✅ Total mask visual check completed successfully!")
            logger.info(f"📁 Check output files in: {args.output_dir}")
            
            if not args.use_option_b:
                logger.info("\n💡 To try shoebox-based masking, add --use-option-b flag")
            
            return 0
        else:
            logger.error("❌ Total mask visual check failed")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    finally:
        close_all_figures()


if __name__ == "__main__":
    sys.exit(main())