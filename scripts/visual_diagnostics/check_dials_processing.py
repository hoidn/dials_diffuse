#!/usr/bin/env python3
"""
Visual diagnostic script for DIALS processing outputs (Module 1.S.1).

This script loads raw still images and their corresponding DIALS processing outputs
(experiments and reflections) and generates visualizations to verify that spot finding,
indexing, and refinement worked correctly.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Add project src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from plot_utils import (
    plot_detector_image,
    plot_spot_overlay,
    ensure_output_dir,
    close_all_figures,
    setup_logging_for_plots,
)

logger = logging.getLogger(__name__)


def load_raw_image(image_path: str):
    """
    Load raw still image using dxtbx.

    Args:
        image_path: Path to raw image file (e.g., CBF)

    Returns:
        ImageSet object from dxtbx
    """
    try:
        from dxtbx.model.experiment_list import ExperimentListFactory
        from dxtbx.imageset import ImageSetFactory

        # Load image as ImageSet
        image_set = ImageSetFactory.new([image_path])[0]
        return image_set

    except ImportError as e:
        logger.error(f"Failed to import dxtbx: {e}")
        logger.info("Install DIALS to use this script with real data")
        return None
    except Exception as e:
        logger.error(f"Failed to load raw image {image_path}: {e}")
        return None


def load_dials_outputs(expt_path: str, refl_path: str):
    """
    Load DIALS experiment and reflection files.

    Args:
        expt_path: Path to .expt file
        refl_path: Path to .refl file

    Returns:
        Tuple of (experiments, reflections) or (None, None) if failed
    """
    try:
        from dxtbx.model.experiment_list import ExperimentListFactory
        from dials.array_family import flex

        # Load experiments
        experiments = ExperimentListFactory.from_json_file(expt_path)

        # Load reflections
        reflections = flex.reflection_table.from_file(refl_path)

        logger.info(
            f"Loaded {len(experiments)} experiments and {len(reflections)} reflections"
        )
        return experiments, reflections

    except ImportError as e:
        logger.error(f"Failed to import DIALS components: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load DIALS outputs: {e}")
        return None, None


def extract_spot_positions(
    reflections, column_name: str = "xyzobs.px.value"
) -> List[Tuple[float, float]]:
    """
    Extract spot positions from reflection table.

    Args:
        reflections: DIALS reflection table
        column_name: Name of column containing xyz pixel coordinates

    Returns:
        List of (x, y) pixel coordinates
    """
    try:
        if not reflections.has_key(column_name):
            logger.warning(f"Reflection table missing column {column_name}")
            return []

        xyz_px = reflections[column_name]
        positions = []

        for i in range(len(xyz_px)):
            x, y, z = xyz_px[i]
            positions.append((x, y))

        logger.info(f"Extracted {len(positions)} spot positions from {column_name}")
        return positions

    except Exception as e:
        logger.error(f"Failed to extract spot positions: {e}")
        return []


def create_dials_visualizations(
    image_set, experiments, reflections, output_dir: str
) -> bool:
    """
    Create visualizations for DIALS processing results.

    Args:
        image_set: Raw image data
        experiments: DIALS experiments
        reflections: DIALS reflections
        output_dir: Directory to save plots

    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = ensure_output_dir(output_dir)

        # Get raw image data (first image, first panel)
        raw_data = image_set.get_raw_data(0)
        if len(raw_data) > 1:
            logger.info(
                f"Multi-panel detector detected ({len(raw_data)} panels), using first panel"
            )
        image_data = raw_data[0]

        # Plot 1: Raw image alone
        logger.info("Creating raw image plot...")
        plot_detector_image(
            image_data,
            title="Raw Still Image",
            output_path=str(output_path / "raw_image.png"),
            log_scale=True,
        )

        # Plot 2: Raw image with observed spots
        logger.info("Creating observed spots overlay...")
        observed_positions = extract_spot_positions(reflections, "xyzobs.px.value")

        if observed_positions:
            plot_spot_overlay(
                image_data,
                observed_positions,
                title="Raw Image with Observed Spots",
                output_path=str(output_path / "image_with_observed_spots.png"),
                log_scale=True,
                spot_color="red",
                spot_size=30,
            )

        # Plot 3: Raw image with predicted spots (if available)
        logger.info("Creating predicted spots overlay...")
        predicted_positions = extract_spot_positions(reflections, "xyzcal.px.value")

        if predicted_positions:
            plot_spot_overlay(
                image_data,
                predicted_positions,
                title="Raw Image with Predicted Spots",
                output_path=str(output_path / "image_with_predicted_spots.png"),
                log_scale=True,
                spot_color="blue",
                spot_size=30,
            )

        # Plot 4: Raw image with both observed and predicted spots
        if observed_positions and predicted_positions:
            logger.info("Creating combined spots overlay...")
            plot_spot_overlay(
                image_data,
                observed_positions,
                title="Raw Image with Observed (red) and Predicted (blue) Spots",
                output_path=str(output_path / "image_with_both_spots.png"),
                log_scale=True,
                spot_color="red",
                spot_size=25,
                predicted_positions=predicted_positions,
                predicted_color="blue",
            )

        # Generate information summary
        info_file = output_path / "dials_processing_info.txt"
        with open(info_file, "w") as f:
            f.write("DIALS Processing Visual Check Results\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Number of experiments: {len(experiments)}\n")
            f.write(f"Number of reflections: {len(reflections)}\n")
            f.write(f"Observed spot positions: {len(observed_positions)}\n")
            f.write(f"Predicted spot positions: {len(predicted_positions)}\n\n")

            if experiments:
                exp = experiments[0]
                f.write("First Experiment Details:\n")
                f.write(
                    f"  Crystal space group: {exp.crystal.get_space_group().info()}\n"
                )
                f.write(f"  Unit cell: {exp.crystal.get_unit_cell()}\n")
                f.write(f"  Detector panels: {len(exp.detector)}\n")
                f.write(f"  Beam wavelength: {exp.beam.get_wavelength():.4f} Å\n")

            if reflections.has_key("partiality"):
                partialities = reflections["partiality"]
                f.write(f"\nPartiality statistics:\n")
                f.write(f"  Mean: {partialities.mean():.3f}\n")
                f.write(f"  Min: {partialities.min():.3f}\n")
                f.write(f"  Max: {partialities.max():.3f}\n")

        logger.info(f"DIALS processing info saved to {info_file}")

        # Suggest DIALS command for reciprocal lattice view
        logger.info("\nFor interactive reciprocal lattice view, run:")
        logger.info(
            f"dials.reciprocal_lattice_viewer {experiments._experiments[0]._filename} {reflections._filename}"
        )

        return True

    except Exception as e:
        logger.error(f"Failed to create DIALS visualizations: {e}")
        return False


def main():
    """Main function for DIALS processing visual checks."""
    parser = argparse.ArgumentParser(
        description="Visual diagnostic for DIALS processing outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check DIALS processing results
  python check_dials_processing.py --raw-image image.cbf --expt processed.expt --refl processed.refl --output-dir dials_check

  # Using existing test data
  python check_dials_processing.py --raw-image ../../747/lys_nitr_10_6_0491.cbf \\
    --expt ../../lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt \\
    --refl ../../lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.refl \\
    --output-dir dials_visual_check
        """,
    )

    parser.add_argument(
        "--raw-image", required=True, help="Path to raw still image file (e.g., CBF)"
    )

    parser.add_argument(
        "--expt", required=True, help="Path to DIALS experiment file (.expt)"
    )

    parser.add_argument(
        "--refl", required=True, help="Path to DIALS reflection file (.refl)"
    )

    parser.add_argument(
        "--output-dir",
        default="dials_visual_check",
        help="Output directory for plots (default: dials_visual_check)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    setup_logging_for_plots()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting DIALS processing visual check...")

    # Validate input files
    for file_path in [args.raw_image, args.expt, args.refl]:
        if not Path(file_path).exists():
            logger.error(f"Input file not found: {file_path}")
            return 1

    try:
        # Load raw image
        logger.info(f"Loading raw image: {args.raw_image}")
        image_set = load_raw_image(args.raw_image)
        if image_set is None:
            logger.error("Failed to load raw image")
            return 1

        # Load DIALS outputs
        logger.info(f"Loading DIALS outputs: {args.expt}, {args.refl}")
        experiments, reflections = load_dials_outputs(args.expt, args.refl)
        if experiments is None or reflections is None:
            logger.error("Failed to load DIALS outputs")
            return 1

        # Create visualizations
        logger.info(f"Creating visualizations in: {args.output_dir}")
        success = create_dials_visualizations(
            image_set, experiments, reflections, args.output_dir
        )

        if success:
            logger.info(f"✅ DIALS processing visual check completed successfully!")
            logger.info(f"📁 Check output files in: {args.output_dir}")
            return 0
        else:
            logger.error("❌ DIALS processing visual check failed")
            return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

    finally:
        close_all_figures()


if __name__ == "__main__":
    sys.exit(main())
