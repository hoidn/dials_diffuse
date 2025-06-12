# DIALS Python API Documentation

This directory contains comprehensive documentation for the DIALS/dxtbx/cctbx Python APIs used in crystallographic data processing.

The documentation has been organized into focused modules for easier navigation and reduced context usage:

## Documentation Files

### [dials_file_io.md](dials_file_io.md)
**File I/O and Model Loading (Sections A.0-A.6)**
- `dials.stills_process` Python API
- Loading experiment lists (`.expt` files)
- Loading reflection tables (`.refl` files)  
- CBF/image file handling with dxtbx
- PHIL parameter management
- File format conversions
- Error handling for I/O operations

### [dxtbx_models.md](dxtbx_models.md)
**Detector, Beam, and Crystal Models (Sections B.0-B.5)**
- Detector geometry and panel access
- Beam properties and wavelength handling
- Crystal unit cell and orientation matrices
- Scan and goniometer models
- Coordinate system transformations
- Multi-panel detector support

### [crystallographic_calculations.md](crystallographic_calculations.md)
**Q-Vector and Crystallographic Calculations (Sections C.1-C.8)**
- Unit cell and space group operations
- Miller index calculations and systematic absences
- Direct Q-vector calculation from detector geometry
- Q-vector validation and comparison methods
- Coordinate transformations (detector ↔ lab ↔ reciprocal space)
- Resolution and d-spacing calculations
- Scattering factor and structure factor calculations
- Thermal motion and B-factor corrections
- Peak integration and background estimation

### [dials_scaling.md](dials_scaling.md)
**DIALS Scaling Framework (Section D.0)**
- Basic scaling workflow and model types
- Physical, KB, and array-based scaling models
- Outlier rejection algorithms
- Cross-validation and quality statistics
- Multi-dataset scaling
- Error model refinement

### [flex_arrays.md](flex_arrays.md)
**Flex Array Operations (Sections D.1-D.4)**
- Basic flex array types and creation
- Mathematical and statistical operations
- Logical operations and masking
- Vector and matrix operations
- Advanced manipulations (concatenation, reshaping, interpolation)
- Custom array functions and utilities

## Usage Notes

- **Version Compatibility**: All documentation is compatible with DIALS 3.x series
- **Import Dependencies**: Each file lists required imports at the beginning
- **Code Examples**: All functions include usage examples and error handling
- **Integration**: Examples show how different modules work together

## Quick Reference

For specific tasks, refer to these sections:

- **Loading DIALS data**: Start with `dials_file_io.md`
- **Detector geometry**: See `dxtbx_models.md` 
- **Q-vector calculations**: Use `crystallographic_calculations.md`
- **Data scaling**: Reference `dials_scaling.md`
- **Array operations**: Check `flex_arrays.md`

## Original File

The original comprehensive documentation is preserved as `DIALS_Python_API_Reference.md` for reference, but the split files above are recommended for daily use.

---

**Note**: This documentation covers the Python APIs needed for implementing diffuse scattering data processing modules. For command-line usage of DIALS tools, refer to the official DIALS documentation at https://dials.github.io/.