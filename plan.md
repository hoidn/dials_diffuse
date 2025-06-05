**`plan.md`**

**Stills Diffuse Scattering Processing Plan (DIALS-Integrated with Integrated Testing)**

**Nomenclature:**

*   `I_raw(px, py, i)`: Raw intensity at detector pixel `(px, py)` for still image `i`. Accessed via the `get_raw_data(0)` method of the `dxtbx.imageset.ImageSet` object corresponding to still `i`.
*   `t_exp(i)`: Exposure time for still `i`. Accessed via the `get_scan().get_exposure_times()[0]` method of the `dxtbx.imageset.ImageSet` object for still `i`, or from image header metadata if a scan object is not present.
*   `Experiment_dials_i`: A `dxtbx.model.experiment_list.ExperimentList` object (typically containing a single `Experiment`) for still `i`, output from `dials.stills_process`. It contains the unique crystal model `Crystal_i` (a `dxtbx.model.Crystal` object).
*   `Reflections_dials_i`: A `dials.array_family.flex.reflection_table` for still `i`, output from `dials.stills_process`. It contains indexed Bragg spots, including partiality values in a column named `"partiality"`.
*   `Mask_pixel`: A global static bad pixel mask (e.g., beamstop, bad pixels, panel gaps), represented as a tuple/list of `dials.array_family.flex.bool` arrays, one per detector panel.
*   `BraggMask_2D_raw_i(px, py)`: A boolean mask for still `i` excluding its own Bragg peak regions, generated from `dials.stills_process` outputs (e.g., via `dials.generate_mask`). Represented as a tuple/list of `dials.array_family.flex.bool` arrays.
*   `Mask_total_2D_i(px, py)`: The combined mask for still `i`, calculated as `Mask_pixel AND (NOT BraggMask_2D_raw_i)`.
*   `v`: Voxel index in the 3D reciprocal space grid defined in Phase 3.
*   `p`: Detector pixel index, typically a tuple `(panel_index, slow_scan_pixel_coord, fast_scan_pixel_coord)`.
*   `|q|`: Magnitude of the scattering vector `q_vector`, calculated as `q_vector.length()`.
*   `H,K,L`: Integer Miller indices, often stored as `dials.array_family.flex.miller_index`.
*   `h,k,l`: Continuous (fractional) Miller indices, resulting from transforming `q_vector` using `Crystal_i.get_A().inverse()` or `Crystal_avg_ref.get_A().inverse()`.
*   `i`: Still image index or a unique identifier for a still image.
*   `P_spot`: Partiality of an observed Bragg reflection, obtained from the `"partiality"` column of `Reflections_dials_i`.

---
**0. Testing Principles and Conventions**

*   **0.1 Granular Testing:** Each significant computational module or processing step defined in this plan will have corresponding tests to verify its correctness and expected outputs based on controlled inputs.
*   **0.2 Input Data Strategy:**
    *   Initial pipeline steps consuming raw still images (Module 1.S.1) will be tested using a small, curated set of real representative still image files (e.g., CBF format). For continuous integration (CI), tiny image stubs (e.g., minimal valid CBF files representing a few detector tiles) or runtime-generated simulated HDF5/CBF-like images will be used to manage repository size and avoid licensing issues.
    *   Subsequent modules will primarily be tested using the serialized output files (e.g., per-still `.expt` files, `.refl` tables, `.pickle` mask files from DIALS; NPZ files from Python components) generated from successfully running tests of preceding modules. This promotes end-to-end validation of data flow and component integration.
    *   For highly focused unit tests of specific algorithms or utility functions *within* a module, inputs may be programmatically constructed in test code if it enhances clarity or allows for more precise control over edge cases.
*   **0.3 Test Data Management:** All test input files (curated CBFs, serialized DIALS outputs, reference outputs, tiny image stubs) will be stored in a dedicated directory within the test suite (e.g., `tests/data/`).
*   **0.4 Verification:** Assertions will check the correctness of calculations, data transformations, and the structure/content of output data structures against pre-calculated expected values or known properties. For modules producing file outputs, tests may compare against reference output files or check key statistics.
*   **0.5 Scope:** Testing will focus on the logic implemented within this project. The internal correctness of external tools like DIALS is assumed, but their integration and invocation by this project (via the adapter layer) will be tested.
*   **0.7 Correction-factor Sign Convention (NEW):**  
    *   **All per-pixel corrections are expressed as *multiplicative* factors**.  
    *   Any divisor returned by an external API (e.g., `Corrections.lp()`) is **immediately inverted** once and stored as a multiplier.  
    *   A helper in `diffusepipe.corrections` (`apply_corrections(raw_I, lp_mult, qe_mult, sa_mult, air_mult)`) centralises this logic and is covered by a regression test using an analytic pixel at 45 °.  The rest of the plan therefore speaks only of multipliers.
*   **0.6 Adapter Layer:** External DIALS/CCTBX/DXTBX Python API calls (e.g., to `dials.stills_process` Python components, scaling framework components, CCTBX utilities) **shall be wrapped** in a thin, project-specific adapter layer residing within the `diffusepipe` package. This adapter layer will be unit-tested against expected behavior based on DIALS/CCTBX documentation and observed outputs, and the main pipeline logic will call these adapters. This localizes changes if the external API evolves and simplifies mocking for higher-level tests.

---

**0.X. Intermediate Quality Control (QC) Checkpoints**

To facilitate early error detection and build confidence in the pipeline's outputs, specific QC metrics and reports will be generated by the `StillsPipelineOrchestrator` after each major data processing phase. These may include simple plots (e.g., saved as PNGs) and text summaries.

*   **After Module 1.S.1 (Per-Still Crystallographic Processing):**
    *   **Metrics/Reports:**
        *   Indexing success rate (percentage of stills successfully indexed).
        *   Distribution of refined unit cell parameters (a, b, c, alpha, beta, gamma) across all indexed stills (histograms, mean, stddev).
        *   Distribution of the number of indexed spots per still.
        *   Distribution of `P_spot` (partiality) values from `Reflections_dials_i`.
        *   RMSD of spot predictions if refinement was performed.
    *   **Purpose:** Assess overall success of DIALS processing, identify problematic stills or batches, evaluate consistency of crystal models.

*   **After Module 2.S.2 (Per-Still Diffuse Intensity Extraction & Correction):**
    *   **Metrics/Reports (for a subset of representative stills):**
        *   Histograms of raw pixel intensities vs. corrected intensities.
        *   2D heatmaps of `TotalCorrection_mult(p)` on the detector.
        *   Q-space coverage plots (e.g., projection of `q_vector` for accepted pixels) for a few sample stills.
        *   Number of pixels passing all filters vs. total unmasked pixels.
    *   **Purpose:** Verify correction factors are reasonable, check filter effectiveness, inspect q-space sampling per still.

*   **After Module 3.S.4 (Merging Relatively Scaled Data into Voxels):**
    *   **Metrics/Reports:**
        *   Overall R-factor or residual statistics from the relative scaling (Module 3.S.3).
        *   Plot of refined scale factors (e.g., `b_i` values vs. `p_order(i)` if applicable).
        *   Redundancy (number of observations per voxel) map/histogram for the `GlobalVoxelGrid`.
        *   Mean intensity vs. resolution for `I_merged_relative`.
    *   **Purpose:** Assess quality and stability of relative scaling, examine data completeness and consistency in the merged dataset.

*   **After Module 4.S.1 (Absolute Scaling and Incoherent Subtraction):**
    *   **Metrics/Reports:**
        *   The determined `Scale_Abs` factor.
        *   Plot of radially averaged experimental total scattering vs. theoretical total scattering (used for determining `Scale_Abs`).
        *   Plot of final `I_abs_diffuse` vs. `|q|`, compared with theoretical coherent scattering `F_calc_sq_voxel`.
        *   Wilson plot of scaled Bragg intensities.
    *   **Purpose:** Validate absolute scale, check appropriateness of incoherent subtraction, overall sanity check of final diffuse map.

Initially, these QC reports (plots and text summaries saved to the output directory) are intended for user review and manual assessment of the pipeline's performance and the quality of intermediate data at each stage. Automatic pipeline interruption or decision-making based on QC metrics can be considered as a future enhancement if specific, robust thresholds and criteria can be defined.

---

**Phase 1: Per-Still Geometry, Indexing, and Initial Masking**

**Module 1.S.1: Per-Still Crystallographic Processing using `dials.stills_process` via Python API**
*   **Action:** For each input still image `i`, use the Python API of `dials.stills_process` (specifically, the `dials.command_line.stills_process.Processor` class via an adapter) to perform spot finding, indexing, optional geometric refinement, and integrate Bragg reflections. This process determines the crystal orientation `U_i`, unit cell `Cell_i`, and reflection partialities `P_spot` for each still.
*   **Input (per still `i` or a small batch of stills):**
    *   File path(s) to raw still image(s).
    *   Base experimental geometry information (`DE_base`), potentially provided as a reference DIALS `.expt` file or constructed programmatically. This includes common detector and source models.
    *   Configuration for `dials.stills_process` (e.g., an instance of `DIALSStillsProcessConfig` from `types_IDL.md`). The adapter layer will translate this into PHIL parameters for `dials.stills_process`. This configuration must ensure:
        *   Calculation and output of reflection partialities (critical).
        *   Saving of shoeboxes if Option B in Module 1.S.3 (Bragg mask from shoeboxes) is chosen.
        *   Appropriate spot finding, indexing, refinement, and integration strategies for the specific still dataset.
        *   Error handling settings (e.g., `squash_errors = False` for debugging).
    *   Known unit cell parameters and space group information can be part of `DIALSStillsProcessConfig` and passed as hints to `dials.stills_process`.
*   **Process (Orchestrated per still `i` or small batch by the `StillsPipelineOrchestrator` component, which calls the adapter for `dials.stills_process.Processor`):**
    To efficiently process datasets containing numerous still images, the `StillsPipelineOrchestrator` **shall implement parallel execution** for this module. This will typically involve distributing the processing of individual stills (or small, independent chunks of stills) across multiple CPU cores, for example, using Python's `multiprocessing.Pool`. Each worker process will execute the full adapter logic for `dials.stills_process.Processor` (points 1-4 below) for its assigned still(s). The orchestrator will be responsible for managing this parallel execution, collecting results (including success/failure status and paths to key output files written to still-specific working directories), and aggregating logs or error messages.
    The subsequent numbered steps (1-4) then describe the logic performed *within each parallel worker* or for each still:
    1.  The adapter initializes a `dials.command_line.stills_process.Processor` instance. It constructs the necessary PHIL parameters from the input `DIALSStillsProcessConfig` (e.g., by merging a base PHIL file with explicitly provided parameters).
    2.  The adapter calls `dials.command_line.stills_process.do_import()` (or equivalent logic within the `Processor`) using the image file path and base geometry to create an initial `dxtbx.model.experiment_list.ExperimentList` for the still.
    3.  The adapter invokes the main processing method of the `Processor` instance (e.g., `processor.process_experiments()`) on the imported experiment(s). This step internally handles:
        *   Spot finding.
        *   Indexing: If multiple lattices are found, `dials.stills_process` typically selects the best based on its internal scoring. If more control is needed, this selection logic might need to be influenced via PHIL parameters or by post-processing its multiple-lattice output if available.
        *   (Optional) Refinement of crystal model `Crystal_i` and experimental geometry.
        *   Integration of indexed Bragg spots, including calculation of partialities.
    4.  The adapter collects the output `integrated_experiments` (a list of `dxtbx.model.Experiment` objects) and `integrated_reflections` (a `dials.array_family.flex.reflection_table`) from the `Processor` instance.
*   **Output (for each successfully processed still `i`):**
    *   `Experiment_dials_i`: The `dxtbx.model.Experiment` object from `integrated_experiments` corresponding to still `i`, containing the refined `Crystal_i`.
    *   `Reflections_dials_i`: A `dials.array_family.flex.reflection_table` (selected from `integrated_reflections` by experiment `id` if `dials.stills_process` outputs a composite table for a batch) containing indexed Bragg spots for still `i`. This table **must** include a column named `"partiality"` containing `P_spot` values. **Note:** The quantitative reliability of `P_spot` values from `dials.stills_process` for true still images requires careful validation. See Module 3.S.3 for strategies on its use in scaling.
    *   (If configured in `dials.stills_process`) Shoebox data associated with `Reflections_dials_i`, stored within the reflection table or as separate files, if needed for Bragg mask generation (Option B in Module 1.S.3).
*   **Consistency Check:** Successful execution of `dials.stills_process` for the still (indicated by the adapter). Validity of the output `Crystal_i` model. Presence and reasonableness of values in the `"partiality"` column of `Reflections_dials_i`.
*   **Testing (Module 1.S.1):**
    *   **Input:** Curated still image file(s), base geometry data, `dials.stills_process` PHIL configuration string.
    *   **Execution:** Invoke the Python adapter for `dials.stills_process.Processor`.
    *   **Verification:** Assert correct generation of `Experiment_dials_i` and `Reflections_dials_i`. `Reflections_dials_i` must contain a `"partiality"` column with valid floating-point numbers. The `Crystal_i` model should be sensible and refined if expected. Test handling of indexing failures for some stills in a batch.

**Module 1.S.2: Static and Dynamic Pixel Mask Generation**
*   **Action:** Create a 2D detector mask `Mask_pixel` based on detector properties, known bad regions (beamstop, panel gaps), and potentially dynamic features (hot/cold/negative pixels) observed across a representative subset of, or all, input stills. This mask is considered global for the dataset.
*   **Input:**
    *   `DE_base.detector`: The base `dxtbx.model.Detector` object.
    *   A representative subset of (or all) raw still images `I_raw(px, py, i)` for dynamic masking.
    *   Configuration parameters defining static masked regions (e.g., beamstop coordinates, untrusted panel/pixel lists).
*   **Process:**
    1.  Generate `Mask_static_panel(px,py)` for each panel using `panel.get_trusted_range_mask(raw_data_for_panel)` (using data from one representative still for the trusted range check if not all stills are processed for this), and by applying masks for beamstop, panel gaps, and user-defined untrusted regions. These are combined using logical AND operations on `flex.bool` arrays.
    2.  Generate `Mask_dynamic_panel(px,py)`: Iterate through the chosen subset of (or all) raw still images. For each pixel, identify if it consistently exhibits anomalous values (e.g., negative counts after pedestal correction, excessively high counts not associated with Bragg peaks). Combine these per-image dynamic masks using logical OR to create a final `Mask_dynamic_panel`.
    3.  `Mask_pixel_panel(px,py) = Mask_static_panel(px,py) AND Mask_dynamic_panel(px,py)`.
        *   **Implementation Note:** All per-pixel operations for dynamic mask generation (e.g., identifying negative, hot, or anomalous pixels across the subset of images) **must be implemented using vectorized array operations** (e.g., leveraging NumPy and DIALS `flex` array capabilities). Python `for` loops iterating over individual pixel indices for full-detector operations are to be avoided due to performance infeasibility.
*   **Output:** `Mask_pixel`: A tuple/list of 2D `dials.array_family.flex.bool` arrays, one per detector panel.
*   **Testing (Module 1.S.2):**
    *   **Input:** Programmatically constructed `dxtbx.model.Detector` object, sample `flex.int` or `flex.double` arrays representing raw pixel data for several "stills" with known anomalous pixels. Static mask configurations.
    *   **Execution:** Call the Python function for static and dynamic mask generation.
    *   **Verification:** Assert that the output `Mask_pixel` correctly identifies known bad regions based on static configurations and correctly flags anomalous pixels based on the dynamic analysis across the provided sample "stills".

**Module 1.S.3: Combined Per-Still Bragg and Pixel Mask Generation**
*   **Action:** For each successfully processed still `i`, generate its specific Bragg peak mask (`BraggMask_2D_raw_i`) using the outputs from Module 1.S.1. Then, combine this with the global `Mask_pixel` (from Module 1.S.2) to produce the final mask (`Mask_total_2D_i`) for diffuse data extraction for that still.
*   **Input (per still `i`):**
    *   `Experiment_dials_i` and `Reflections_dials_i` (from Module 1.S.1).
    *   (If Option B is chosen) Shoebox data associated with `Reflections_dials_i`.
    *   `Mask_pixel` (from Module 1.S.2).
    *   Configuration parameters for `dials.generate_mask` (if Option A is chosen) or for custom mask generation logic from shoeboxes.
*   **Process (per still `i`):**
    1.  **Generate `BraggMask_2D_raw_i`:**
        *   **Option A (Recommended):** Execute `dials.generate_mask` (via its Python API adapter) using `Experiment_dials_i` and `Reflections_dials_i` as input. Configure `dials.generate_mask` to create a mask that covers the regions occupied by these indexed Bragg spots.
        *   **Option B (Alternative, if shoeboxes are preferred):** If `dials.stills_process` was configured to save shoeboxes with foreground/background mask information, extract the foreground mask for each reflection in `Reflections_dials_i` and project/combine these onto a 2D per-panel mask to form `BraggMask_2D_raw_i`. This projection should use pixels flagged with `dials.algorithms.shoebox.MaskCode.Foreground` and/or `dials.algorithms.shoebox.MaskCode.Strong` to define Bragg regions, rather than relying on potentially noisy `P_spot` values for this masking purpose.
    2.  Combine with static/dynamic pixel mask: `Mask_total_2D_i(px,py) = Mask_pixel(px,py) AND (NOT BraggMask_2D_raw_i(px,py))`. The logical NOT inverts `BraggMask_2D_raw_i` so that Bragg regions are `false`.
        **Data Flow for Masks:**
            1.  The `StillsPipelineOrchestrator` maintains the global `Mask_pixel` (from Module 1.S.2).
            2.  For the current still `i`, the `StillsPipelineOrchestrator` generates a temporary `BraggMask_2D_raw_i` (using logic from this module).
            3.  The `StillsPipelineOrchestrator` then computes the temporary `Mask_total_2D_i = Mask_pixel AND (NOT BraggMask_2D_raw_i)`.
            4.  This `Mask_total_2D_i` object (e.g., tuple of `flex.bool` arrays) is passed directly as an argument to the `DataExtractor` (for Modules 2.S.1 & 2.S.2) along with the raw image data for still `i`.
            5.  After the `DataExtractor` has processed still `i`, the temporary `BraggMask_2D_raw_i` and `Mask_total_2D_i` for that still can be discarded by the `StillsPipelineOrchestrator` to conserve memory.
            This streamed approach ensures that only one (or a small batch) of complete per-still total masks needs to be in memory at any given time.
*   **Output (per still `i`):** `Mask_total_2D_i` (a tuple/list of `dials.array_family.flex.bool` arrays, one per panel).
*   **Testing (Module 1.S.3):**
    *   **Input:** Sample `Experiment_dials_i`, `Reflections_dials_i` (with known spot positions), and `Mask_pixel`. Configuration for mask generation.
    *   **Execution:** Call the adapter for `dials.generate_mask` or the shoebox processing logic, then combine the resulting Bragg mask with `Mask_pixel`.
    *   **Verification:** Assert that `Mask_total_2D_i` correctly excludes the known Bragg peak regions for that specific still's orientation and reflection list, and also incorporates the exclusions from `Mask_pixel`.

---

**Phase 2: Per-Still Diffuse Intensity Extraction and Pixel-Level Correction**
*(Corresponds to the `DataExtractor` IDL interface)*

**Module 2.S.1: Pixel-Based Diffuse Data Extraction & Q-Calculation**
*   **Action:** For each still `i`, iterate through its detector pixels. If a pixel is deemed suitable for diffuse scattering analysis (i.e., it passes `Mask_total_2D_i`), extract its raw intensity and calculate its corresponding q-vector using the specific geometry of still `i`.
*   **Input (per still `i`):**
    *   Raw image data for still `i` (accessible via an `dxtbx.imageset.ImageSet` object for still `i`).
    *   `Experiment_dials_i` (containing `Detector_i`, `Beam_i`, and `Crystal_i` from Module 1.S.1).
    *   `Mask_total_2D_i` (from Module 1.S.3).
    *   Pixel step configuration (e.g., process every Nth pixel).
*   **Process (for each pixel `p=(panel_idx, py, px)` on still `i`, respecting pixel step):**
    1.  If `Mask_total_2D_i[panel_idx](py, px)` is `true`:
        a.  `I_raw_val = ImageSet_i.get_raw_data(0)[panel_idx][py, px]`. (The `0` index assumes the ImageSet for a still contains one image).
        b.  Calculate `q_vector(p)` using `Experiment_dials_i.detector[panel_idx]` and `Experiment_dials_i.beam`. The crystal model `Experiment_dials_i.crystal` is implicitly used by these detector/beam models if they were refined together. If `Experiment_dials_i.crystal.num_scan_points > 1` (unusual for a single still from `stills_process` but possible if it represents a micro-sweep), ensure `get_A_at_scan_point(0)` is used.
        c.  Store the tuple: `(q_vector(p), I_raw_val, panel_idx, px, py, still_identifier=i)`.
*   **Output (per still `i`):**
    *   `RawDiffusePixelList_i`: A list of tuples, where each tuple is `(q_vector, raw_intensity, original_panel_index, original_fast_pixel_coord, original_slow_pixel_coord, still_identifier)`. `q_vector` is a `scitbx.matrix.col` object.
    *   **Implementation Note on Performance:** If iterating through the *unmasked* pixels (after `Mask_total_2D_i` is applied) still results in a very large number of pixels per still, direct Python looping might be a bottleneck. In such cases, the implementation should first identify the coordinates of all `True` pixels in `Mask_total_2D_i` (e.g., `np.where(mask_numpy)`). These coordinates can then be passed as batches to efficient DXTBX/CCTBX functions for q-vector calculation (e.g., `panel.get_lab_coord_multiple()`).
*   **Testing (Module 2.S.1):**
    *   **Input:** Sample still image data (e.g., a small `flex` array), a corresponding `Experiment_dials_i` object, and a `Mask_total_2D_i`.
    *   **Execution:** Call the Python function(s) implementing this module.
    *   **Verification:** Assert that `RawDiffusePixelList_i` contains the correct q-vectors (calculated using `scitbx.matrix.col`) and raw intensity values for the expected unmasked pixels.

**Module 2.S.2: Pixel-Based Correction Factor Application**
*   **Action:** For each diffuse pixel extracted in Module 2.S.1, calculate and apply all relevant geometric and experimental correction factors to its intensity, and propagate uncertainties.
*   **Input (per still `i`):**
    *   `RawDiffusePixelList_i` (from Module 2.S.1).
    *   `Experiment_dials_i` (for geometry models: `Detector_i`, `Beam_i`, `Goniometer_i` if relevant).
    *   `t_exp(i)` (exposure time for still `i`).
    *   Detector gain value.
    *   Configuration parameters for all applicable corrections (e.g., detector thickness/material for efficiency, air path details if not derived from geometry).
    *   Background subtraction parameters (e.g., path to a pre-processed background map for still `i` or a relevant group, or a constant background value).
    *   Configuration for resolution and intensity filters.
*   **Process (for each entry `(q_vec, I_raw_val, panel_idx, px, py, still_id)` in `RawDiffusePixelList_i`):**
    1.  **Calculate Per-Pixel Correction Factors `TotalCorrection_mult(p)` Using DIALS Corrections API:**
        *   All individual correction factors **shall be converted to and combined as multipliers** to form `TotalCorrection_mult(p)`. The adapter layer or calculation logic for each specific correction type is responsible for this conversion.
        *   **DIALS Corrections Object Setup:** For each still `i`, the `DataExtractor` (or its adapter layer) will instantiate a `dials.algorithms.integration.Corrections` object using `Experiment_dials_i.beam`, `Experiment_dials_i.goniometer` (if present), and `Experiment_dials_i.detector`. This object provides robust, well-tested implementations for standard geometric corrections.
        *   **For the array of accepted diffuse pixel observations** (defined by their s1 vectors and panel IDs for still `i`):
            *   **Lorentz-Polarization (LP) Correction:** `LP_divisors` will be obtained by calling the `lp()` method of the DIALS `Corrections` object. Convert to multiplier: `LP_mult(p) = 1.0 / LP_divisor(p)`.
            *   **Detector Quantum Efficiency (QE) Correction:** `QE_multipliers` will be obtained by calling the `qe()` method of the DIALS `Corrections` object. Use directly: `QE_mult(p) = QE_multiplier(p)`.
            *   **Note:** The DIALS `Corrections` object handles complex effects like parallax internally for LP and QE corrections.
        *   **Custom Corrections (Not Available from DIALS Corrections for Arbitrary Diffuse Pixels):**
            *   **Solid Angle Correction:** The plan must retain a custom (but carefully validated) calculation for `SolidAngle_divisor(p)` based on `Experiment_dials_i.detector[panel_idx]` geometry and pixel coordinates, since diffuse pixels are not necessarily at Bragg positions where DIALS corrections are typically applied. Convert to multiplier: `SA_mult(p) = 1.0 / SolidAngle_divisor(p)`.
            *   **Air Attenuation Correction:** Retain a custom calculation for `AirAttenuation_divisor(p)` based on path length from sample to pixel and X-ray wavelength. Convert to multiplier: `Air_mult(p) = 1.0 / AirAttenuation_divisor(p)`.
        *   **Final Assembly:** `TotalCorrection_mult(p) = LP_mult(p) × QE_mult(p) × SA_mult(p) × Air_mult(p)`.  
          All four terms are already in multiplier form because of rule 0.7.
        *   **Regression Test Requirement:** A regression test must be implemented that, for a synthetic experiment and a few selected pixel positions (including off-Bragg positions), compares the individual `LP_divisor` and `QE_multiplier` values obtained from the DIALS `Corrections` adapter against trusted reference values or a separate, careful analytic calculation for those specific points. The custom Solid Angle and Air Attenuation calculations must also have dedicated unit tests with known geometric configurations.
    2.  **Background Subtraction:**
        *   If a background map relevant to still `i` is provided: Load map, retrieve value `I_bkg_map(p)` for the current pixel, and subtract it from `I_raw_val`. The variance of the background map value `Var_bkg(p)` must also be retrieved.
        *   If a constant background value `BG_const` is configured: `I_raw_val_bg_sub = I_raw_val - BG_const`. `Var_bkg(p)` is typically 0 unless `BG_const` has an associated uncertainty.
        *   If no background subtraction: `I_raw_val_bg_sub = I_raw_val`. `Var_bkg(p)` is 0.
    3.  **Apply Gain and Exposure Time Normalization:**
        *   `I_processed_per_sec = I_raw_val_bg_sub / t_exp(i)`.
        *   `Var_photon_initial = I_raw_val / gain` (variance of the original raw count before background subtraction, assuming Poisson statistics).
        *   `Var_processed_per_sec = (Var_photon_initial + Var_bkg(p)) / (t_exp(i))^2`.
    4.  **Apply Total Multiplicative Correction:**
        *   `I_corrected_val = I_processed_per_sec * TotalCorrection_mult(p)`.
    5.  **Error Propagation:**
        *   `Var_corrected_val = Var_processed_per_sec * (TotalCorrection_mult(p))^2`. This assumes `TotalCorrection_mult(p)` has negligible uncertainty. If uncertainties of correction factors are significant and known, they should be propagated here. This simplification (ignoring correction factor uncertainties) must be documented.
        *   `Sigma_corrected_val = sqrt(Var_corrected_val)`.
    6.  **Resolution and Intensity Filtering:**
        *   Calculate d-spacing from `q_vec.length()`.
        *   If `I_corrected_val` or its d-spacing fall outside the configured minimum/maximum intensity or resolution limits, discard this pixel observation.
*   **Output (per still `i`):**
    *   `CorrectedDiffusePixelList_i`: A list of tuples `(q_vector, I_corrected, Sigma_corrected, q_x, q_y, q_z, original_panel_index, original_fast_pixel_coord, original_slow_pixel_coord, still_identifier)` for all pixels that passed all filters. `q_x, q_y, q_z` are components of `q_vector`.
*   **Testing (Module 2.S.2):**
    *   **Input:** Sample `RawDiffusePixelList_i`, a corresponding `Experiment_dials_i` object, exposure time, gain, and configurations for all corrections and filters.
    *   **Execution:** Call the Python function(s) implementing this correction module.
    *   **Verification:** Assert that `I_corrected` and `Sigma_corrected` in `CorrectedDiffusePixelList_i` match expected values based on known correction formulas and inputs. Verify adherence to the "all multipliers" convention for corrections. Test filter logic.
    *   **DIALS Corrections Regression Test:** For a synthetic experiment, verify that `apply_corrections()` recovers the analytic intensity of a 45 ° pixel to <1 % once LP has been inverted.
    *   **Custom Corrections Unit Tests:** The custom Solid Angle and Air Attenuation calculations must have dedicated unit tests with known geometric configurations and expected theoretical values.
    *   **Validation Test for Combined Corrections:** Include a specific unit test that provides synthetic pixel data with known, simple geometric properties. This test should calculate `TotalCorrection_mult(p)` using the implemented logic (including calls to adapters for DIALS corrections and custom calculations) and assert that the final combined multiplicative factor matches a pre-calculated, theoretically correct value. This validates both the individual correction factor calculations/conversions and their final combination.

---

**Implementation Note on Efficient Pixel Data Handling (Relevant to Modules 2.S.1, 2.S.2, and 3.S.2)**

While the plan may describe intermediate outputs like `RawDiffusePixelList_i` or `CorrectedDiffusePixelList_i` as "lists of tuples" for conceptual clarity of per-pixel transformations, the actual implementation **must** prioritize memory efficiency and processing speed by using array-based data structures and vectorized operations.

1.  **Per-Still Data Storage (Output of Module 2.S.2):**
    *   Instead of a Python list of individual pixel tuples for each still, the data for accepted diffuse observations from still `i` (i.e., the conceptual `CorrectedDiffusePixelList_i`) should be stored as a set of synchronized, contiguous arrays. For example:
        *   `final_q_vectors_still_i`: An array of shape (M, 3) for q-vectors (where M is the number of accepted pixels for still `i`).
        *   `final_I_corrected_still_i`: An array of shape (M) for corrected intensities.
        *   `final_Sigma_corrected_still_i`: An array of shape (M) for corresponding sigmas.
        *   (Optional but recommended) `final_original_pixel_coords_still_i`: An array of shape (M, 3) storing `(panel_idx, fast_coord, slow_coord)`.
        *   (Optional but recommended) `final_q_components_still_i`: An array of shape (M,3) for `(qx, qy, qz)`.
    *   These arrays should be NumPy arrays or DIALS `flex` arrays.

2.  **Vectorized Processing (Modules 2.S.1, 2.S.2):**
    *   All steps involving per-pixel calculations—such as applying masks, calculating q-vectors for multiple pixels, applying geometric/experimental corrections, and filtering by intensity or resolution—**must be implemented using vectorized operations** on these arrays.
    *   For example, in Module 2.S.1, after applying `Mask_total_2D_i` to a panel's raw data, obtain the coordinates of accepted pixels (e.g., via `np.where` or `flex.bool.iselection()`) and pass these coordinate arrays to batch-capable functions like `panel.get_lab_coord_multiple()` for efficient q-vector calculation.
    *   Similarly, in Module 2.S.2, all correction factors should be calculated as arrays corresponding to the accepted pixels, and then applied simultaneously through array arithmetic.

3.  **Data Handling for Binning and Scaling (Modules 3.S.2 and 3.S.3):**
    *   The per-still array sets (output of Module 2.S.2, as described in point 1 of this note) are the input to Module 3.S.2 (Binning).
    *   **For a first-pass implementation, the `BinnedPixelData_Global` structure created in Module 3.S.2 will be assumed to be manageable in memory.** This structure will map each `voxel_idx` in the `GlobalVoxelGrid` to a collection of all `(I_corrected, Sigma_corrected, still_id, ...other_relevant_per_observation_data...)` tuples/objects that fall into that voxel from across all processed stills.
    *   This in-memory `BinnedPixelData_Global` will then be directly used by Module 3.S.3 (Relative Scaling) for reference generation and residual calculation.
    *   **Future Optimization:** If memory limitations are encountered with very large datasets (i.e., the total number of accepted diffuse pixel observations across all stills is too large to hold all associated `I_corrected, Sigma_corrected, still_id` information in RAM, even when grouped by voxel), this strategy will need to be revisited. Future optimizations might include:
        *   Saving per-still accepted pixel arrays to disk and reading them sequentially during scaling.
        *   Implementing more complex streamed or on-disk aggregation for `BinnedPixelData_Global` if direct in-memory lists per voxel become too large.
    *   The initial implementation should proceed with the assumption of in-memory viability for `BinnedPixelData_Global` to simplify the first pass of development for the scaling algorithms. Performance and memory profiling will guide the need for these future optimizations.

This array-centric approach is critical for achieving acceptable performance and managing memory effectively for large-scale still diffuse scattering datasets.

---

**Phase 3: Voxelization, Relative Scaling, and Merging of Diffuse Data**

**Module 3.S.1: Global Voxel Grid Definition**
*   **Action:** Define a common 3D reciprocal space grid that will be used for merging diffuse scattering data from all processed still images.
*   **Input:**
    *   A collection of all `Experiment_dials_i` objects (specifically their `Crystal_i` models) from Phase 1.
    *   A collection of all `CorrectedDiffusePixelList_i` (from Phase 2) to determine the overall q-space coverage and thus the HKL range.
    *   Configuration parameters for the grid: target resolution limits (`d_min_target`, `d_max_target`), number of divisions per reciprocal lattice unit (e.g., `ndiv_h, ndiv_k, ndiv_l`), and optionally a specific reference unit cell if not averaging.
*   **Process:**
    1.  Determine an average reference crystal model (`Crystal_avg_ref`) from all `Crystal_i` models.  
       After averaging, compute the rms Δhkl for a sample of Bragg reflections transformed with `A_ref⁻¹`.  
       *If rms Δhkl > 0.1* the per-still orientation matrix `U_i` is used to pre-rotate each `q_vec` before the common transform (see Module 3.S.2).
        This involves:
        a.  **Average Unit Cell (`UC_avg`):** Robustly average all `Crystal_i.get_unit_cell()` parameters using CCTBX utilities to obtain `UC_avg`. This defines the reciprocal metric tensor `B_avg_ref = (UC_avg.fractionalization_matrix())^-T`.
        b.  **Average Orientation Matrix (`U_avg_ref`):** Extract `U_i = Crystal_i.get_U()` for each still. Average these `U_i` matrices using a robust method suitable for rotation matrices (e.g., conversion to quaternions, averaging, and conversion back to a matrix; or averaging of small rotation vectors if deviations are minimal). The chosen method must be documented in the implementation.
        c.  The final setting matrix for the grid is `A_avg_ref = U_avg_ref * B_avg_ref`. This matrix defines the grid's orientation and cell in the common laboratory frame.
    2.  Using `Crystal_avg_ref`, transform all `q_vector` components from all `CorrectedDiffusePixelList_i` to fractional Miller indices `(h,k,l)` to determine the overall minimum and maximum H, K, L ranges covered by the data, considering the target resolution limits.
    3.  Define the `GlobalVoxelGrid` object. This object will store:
        *   `Crystal_avg_ref` (the `dxtbx.model.Crystal` object used for HKL transformations to/from the grid).
        *   The integer HKL range of the grid.
        *   The number of subdivisions per unit cell (`ndiv_h, ndiv_k, ndiv_l`).
        *   Methods for converting `(h,k,l)` to `voxel_idx` and vice-versa.
*   **Output:** `GlobalVoxelGrid` (a single grid definition object for the entire dataset).
*   **Testing (Module 3.S.1):**
    *   **Input:** A list of minimal `Experiment_dials_i` objects (programmatically constructed, containing `Crystal_i` models) and a sample `CorrectedDiffusePixelList_i` (with `scitbx.matrix.col` q-vectors) to define HKL ranges. Grid definition parameters.
    *   **Execution:** Call the grid definition function.
    *   **Verification:** Assert that the `GlobalVoxelGrid` object has the correct reference crystal model, HKL range, and voxel divisions based on the inputs.

**Module 3.S.2: Binning Corrected Diffuse Pixels into Global Voxel Grid**
*   **Action:** Assign each corrected diffuse pixel observation from all stills to its corresponding voxel in the `GlobalVoxelGrid`, applying Laue group symmetry. Implement streamed voxel accumulation to manage memory.
*   **Input:**
    *   A collection of all `CorrectedDiffusePixelList_i` from Phase 2.
    *   `GlobalVoxelGrid` (from Module 3.S.1, containing `Crystal_avg_ref`).
    *   Laue group symmetry information (e.g., space group string or `cctbx.sgtbx.space_group` object).
*   **Process (Memory Management: This process must use streamed voxel accumulation, e.g., Welford's algorithm or similar, to update voxel statistics incrementally without holding all pixel data in memory simultaneously):**
    0.  **Memory forecasting:** Run a quick pass over a 1 % subset of pixels to estimate bytes / pixel.  
       If the projected full set would exceed 30 % of available RAM, switch automatically to an HDF5-backed chunked store (`VoxelAccumulator`) that streams observations voxel-by-voxel.  
    1.  Initialise either an in-memory accumulator (small jobs) **or** an on-disk `VoxelAccumulator` (large jobs) that keeps only one voxel’s running stats in RAM at a time.
    2.  For each still `i` and each observation `(q_vec, I_corr, Sigma_corr, ..., still_id)` in its `CorrectedDiffusePixelList_i`:
        a.  Transform `q_vec` to fractional indices:  
           `hkl_frac = A_ref⁻¹ · (U_iᵀ · q_vec)` **if per-still orientation correction is required** (see test in Module 3.S.1); otherwise omit `U_iᵀ`.
            **Note on Transformation:** This transformation directly maps the lab-frame q-vector from an individual still `i` into the fractional HKL coordinate system defined by the average reference crystal (`Crystal_avg_ref`) and its setting matrix `A_avg_ref`. No intermediate rotation of `q_lab_i` is required before this step when using this direct transformation.
        b.  Map `(h,k,l)` to the asymmetric unit of the specified Laue group using CCTBX symmetry functions (e.g., via an adapter for `space_group.info().map_to_asu()`), obtaining `(h_asu, k_asu, l_asu)`.
        c.  Determine the `voxel_idx` in `GlobalVoxelGrid` corresponding to `(h_asu, k_asu, l_asu)`.
        d.  Update the accumulation data structure for `voxel_idx` with `(I_corr, Sigma_corr, still_id)`. If using Welford's, update mean, M2, and count for that `still_id` within that voxel, or for the voxel globally if scales are to be applied before final merge.
*   **Output:**
    *   `BinnedPixelData_Global`: A data structure (e.g., a dictionary mapping `voxel_idx` to a list of `(I_corr, Sigma_corr, still_id)` observations, or to aggregated statistics if full streaming is done). This structure holds all observations grouped by their target voxel in the asymmetric unit.
    *   `ScalingModel_initial_list`: A list of initial scaling model parameter objects, one for each still `i` (or group of stills if a grouping strategy is employed). Parameters typically initialized to unity scale and zero offset.
*   **Testing (Module 3.S.2):**
    *   **Input:** A sample `CorrectedDiffusePixelList_i` (with known `q_vector`s and values), a `GlobalVoxelGrid` object, and Laue group information.
    *   **Execution:** Call the binning and accumulation function.
    *   **Verification:** Assert that observations are correctly assigned to `voxel_idx` after HKL transformation and ASU mapping. Verify that accumulated statistics (if using Welford's) are correct for sample voxels.

**Module 3.S.3: Relative Scaling of Binned Observations (Custom Implementation using DIALS Components)**
*   **Action:** Iteratively refine scaling model parameters for each still `i` (or group of stills), using a custom-defined diffuse scaling model. This model will be built using DIALS/CCTBX components for parameterization (e.g., `SingleScaleFactor`, `GaussianSmoother`) and refinement (e.g., DIALS parameter manager, minimizers like Levenberg-Marquardt).
*   **Input:**
    *   `BinnedPixelData_Global` (from Module 3.S.2).
    *   `ScalingModel_initial_list` (from Module 3.S.2).
    *   Configuration for the custom diffuse scaling model. **Note on Iterative Model Development and Configuration:**
        1.  **Initial v1 Model (parameter-guarded):**  
            *   Exactly **one** free parameter per still: global multiplier `b_i`.  
            *   Optional **single 1-D resolution smoother** with ≤ 5 control points shared by all stills (`a(|q|)`), enabled only if `enable_res_smoother = True` in config.  
            *   **Panel, spatial or additive terms are *hard-disabled* in v1.** Attempting to enable them raises a configuration error.  
            *   A global constant `MAX_FREE_PARAMS = 5 + N_stills` is enforced; CI fails if exceeded.
        2.  **Developer-Led Model Evolution:** The process of adding more complexity to the active scaling model is primarily a developer-led activity during pipeline validation and refinement:
            *   After successfully running and testing the pipeline with the initial simple model, analyze scaling residuals for systematic trends (e.g., versus detector position `px,py`, panel ID, or the still ordering parameter `p_order(i)`).
            *   If significant, smooth, and interpretable trends are observed, the developer will incrementally enable and configure more complex components (e.g., `d_i(panel)` if strong panel-to-panel variations persist; `a_i(px,py,p_order(i))` using `GaussianSmoother2D/3D` if warranted by clear spatial residuals).
            *   Each newly activated component and its parameterization (e.g., number of bins, smoother parameters) must be validated for stability and its impact on improving the fit and data quality.
            *   **Critical for Additive Terms:** Additive offset components (like `c_i(|q|)` for background) should only be introduced after a stable multiplicative scaling solution is achieved, and ideally with strong physical justification or clear evidence from residuals. If both multiplicative and additive terms are refined simultaneously in later iterations, careful attention must be paid to parameter constraints, regularization, and potential correlations, as these terms can trade off leading to non-unique solutions.
        3.  **User-Facing Configuration:** The `advanced_components` section is ignored until a residual plot proves need; enabling it requires `--experimental-scale-model` CLI flag.
    *   The implementation will use DIALS components like `dials.algorithms.scaling.model.components.scale_components.ScaleComponentBase`, `SingleScaleFactor`, and `GaussianSmoother1D/2D/3D` as appropriate for the chosen parameterizations.
    *   Definition and source of the still ordering parameter `p_order(i)` must be configurable if components using it (like smoothers dependent on `p_order(i)`) are activated.
    *   A collection of all `Reflections_dials_i` (from Module 1.S.1), containing `I_bragg_obs_spot` and the `"partiality"` column (`P_spot`), for use in Bragg-based reference generation. **Critical Partiality Handling Strategy:** 
        *   **Partiality handling strategy (revised):**  
            *   `P_spot` **is *only* used as a quality flag**.  We keep reflections with `P_spot ≥ P_min_thresh` (default 0.1) but we **never divide by it** at any later stage.  
            *   Absolute scale will be obtained from Wilson statistics on merged Bragg data rather than from partiality-corrected intensities.  
            *   Unit tests formerly referring to “divide by P_spot” are removed; new tests assert that intensity values are unchanged after the quality filter.
*   **Process (Iterative, using a minimal‐parameter v1 model):**
    1.  **Parameter Management Setup:** Initialize DIALS's active parameter manager with all refineable parameters from all `ScalingModel_i` components. Set up any restraints (e.g., for smoothness of Gaussian smoother parameters).
    2.  **Iterative Refinement Loop:**
        a.  **Reference Generation:**
            *   **Bragg Reference (if used):** For each unique HKL (mapped to ASU), calculate a reference intensity `I_bragg_merged_ref(HKL_asu)`. This is a weighted average of `(I_bragg_obs_spot / P_spot)` from all contributing stills, where each term is divided by the current estimate of its still's multiplicative scale factors (from the current `ScalingModel_i`). `P_spot` is taken from the `"partiality"` column.
            *   **Diffuse Reference:** For each `voxel_idx`, calculate a reference intensity `I_diffuse_merged_ref(voxel_idx)`. This is a weighted average of `(I_diffuse_obs_for_still_i - current_additive_offset_for_still_i)` from all contributing stills, where each term is divided by the current estimate of its still's multiplicative scale factors.
        b.  **Scaling Model Parameter Refinement:**
            *   Instantiate the overall custom `DiffuseScalingModel` which aggregates all per-still/group `ScalingModel_i` components.
            *   Use a DIALS-compatible minimizer (e.g., Levenberg-Marquardt or Gauss-Newton, accessed via the adapter layer) with the custom `DiffuseScalingTarget` function.
            *   The `DiffuseScalingTarget` function calculates residuals for each observation in `BinnedPixelData_Global`. For an observation `I_obs` from still `i` in voxel `v`:
                `Residual_v,i = ( (I_obs - c_i_model_value) / (b_i_model_value * d_i_model_value * a_i_model_value) ) - I_reference(v)`.
                (The exact form depends on how `a,b,c,d` components are defined as additive or multiplicative).
            *   The minimizer adjusts the refineable parameters (e.g., control points of Gaussian smoothers, single scale factors) to minimize the sum of squared, weighted residuals.
        c.  **Convergence Check:** Evaluate convergence criteria (e.g., change in R-factor, change in parameters). If not converged, repeat from step 2a.
*   **Output:** `ScalingModel_refined_list` (containing the refined parameters for the custom `DiffuseScalingModel` components for each still or group).
*   **Testing (Module 3.S.3):**
    *   **Input:** `BinnedPixelData_Global`, `Reflections_dials_i` (with `"partiality"` column), initial scaling model parameters, and configurations for the custom scaling model components (including functional forms and parameterization strategies).
    *   **Execution:** Run the custom relative scaling procedure.
    *   **Verification:** Test with synthetic `BinnedPixelData_Global` where known scale factors, offsets, and sensitivity variations have been applied to different "stills." Assert that the refined parameters in `ScalingModel_refined_list` recover these known variations. Verify correct use of `P_spot` from the `"partiality"` column in Bragg reference generation. Check for convergence of the minimizer and reasonable residual values.

**Module 3.S.4: Merging Relatively Scaled Data into Voxels**
*   **Action:** Apply the final refined scaling model parameters (from `ScalingModel_refined_list`) to all observations in `BinnedPixelData_Global` and merge them within each voxel of the `GlobalVoxelGrid`.
*   **Input:**
    *   `BinnedPixelData_Global` (containing per-voxel lists of `(I_corr, Sigma_corr, still_id)`).
    *   `ScalingModel_refined_list` (containing parameters for the custom `DiffuseScalingModel` components for each still/group).
    *   `GlobalVoxelGrid`.
*   **Process:**
    1.  For each observation `(I_corr, Sigma_corr, still_id, ...)` associated with a `voxel_idx` in `BinnedPixelData_Global`:
        a.  Retrieve the refined `ScalingModel_i` (or its components) for the corresponding `still_id`.
        b.  Calculate the total multiplicative scale `M_i` and total additive offset `C_i` for this observation using the refined model parameters and the observation's specific properties (e.g., `|q|`, `px,py`, `panel_idx`, `p_order(i)`).
        c.  Apply the scaling: `I_final_relative = (I_corr - C_i) / M_i`.
        d.  Propagate uncertainties: `Sigma_final_relative = Sigma_corr / abs(M_i)` (simplification, assuming `M_i` and `C_i` have negligible uncertainty compared to `Sigma_corr`).
    2.  For each `voxel_idx` in `GlobalVoxelGrid`:
        a.  Collect all `I_final_relative` and `Sigma_final_relative` values from observations binned to this `voxel_idx`.
        b.  Perform a weighted merge (typically inverse variance weighting: `weight = 1 / Sigma_final_relative^2`) to calculate `I_merged_relative(voxel_idx)` and `Sigma_merged_relative(voxel_idx)`.
        c.  Store `num_observations_in_voxel(voxel_idx)`.
        d.  Calculate `q_center_x, q_center_y, q_center_z` and `|q|_center` for the voxel using `GlobalVoxelGrid.Crystal_avg_ref`.
*   **Output:** `VoxelData_relative`: A data structure (e.g., a `flex.reflection_table` or NumPy structured array) where each row represents a voxel and contains `(voxel_idx, H_center, K_center, L_center, q_center_x, q_center_y, q_center_z, |q|_center, I_merged_relative, Sigma_merged_relative, num_observations_in_voxel)`.
*   **Testing (Module 3.S.4):**
    *   **Input:** Sample `BinnedPixelData_Global`, a `ScalingModel_refined_list` with known refined parameters, and `GlobalVoxelGrid`.
    *   **Execution:** Call the merging function.
    *   **Verification:** Assert that `I_merged_relative` and `Sigma_merged_relative` in `VoxelData_relative` are correct based on the applied scales and weighted averaging logic.

---

**Phase 4: Absolute Scaling**

**Module 4.S.1: Absolute Scaling and Incoherent Scattering Subtraction**
*   **Action:** Convert the relatively scaled diffuse map (`VoxelData_relative`) to absolute units (e.g., electron units per unit cell) and subtract the theoretically calculated incoherent scattering contribution.
*   **Input:**
    *   `VoxelData_relative` (from Module 3.S.4).
    *   `unitCellInventory`: The complete atomic composition of the unit cell (e.g., a dictionary of element symbols to counts).
    *   `GlobalVoxelGrid.Crystal_avg_ref` (the `dxtbx.model.Crystal` object defining the unit cell and symmetry for theoretical calculations).
    *   Merged, relatively-scaled Bragg intensities **with no partiality correction**.  Scale derivation will follow a Wilson-style fit of ⟨|F|²⟩ vs resolution; reflections fail the quality filter if `P_spot < P_min_thresh`.
*   **Process:**
    1.  **Theoretical Scattering Calculation:**
        *   For each unique element in `unitCellInventory`, obtain its atomic form factor `f0(element, |q|)` and its **full incoherent (Compton) scattering cross-section** `S_incoh(element, |q|)`. This calculation will be performed via an adapter layer for CCTBX scattering utilities. **Note on Q-Range for Incoherent Scattering:** The adapter layer is responsible for ensuring the accuracy of `S_incoh(element, |q|)` across the entire q-range relevant to the diffuse data. This typically involves a hybrid strategy:
            1.  Utilize reliable tabulated data from CCTBX (e.g., IT1992 data accessible via `cctbx.eltbx.sasaki.table().incoherent()`) for q-ranges where these tables are accurate and valid (generally up to `|q| \approx 4 Å⁻¹`).
            2.  For `|q|` values exceeding the valid range of the tabulated data, the adapter must supplement or switch to an appropriate analytical formulation for high-q X-ray incoherent scattering. Priority should be given to using routines from `cctbx.eltbx.formulae` if they provide accurate high-q incoherent cross-sections. If not, the adapter should implement or utilize a well-established formula such as the relativistic Klein-Nishina formula.
            The source (table/formula) and any approximations used for calculating `S_incoh` across different q-ranges must be clearly documented within the adapter's implementation.
        *   For each voxel in `VoxelData_relative` (with its `|q|_center`):
            *   Calculate `F_calc_sq_voxel = sum_atoms_in_UC (f0(atom, |q|_center)^2)`.
            *   Calculate `I_incoherent_theoretical_voxel = sum_atoms_in_UC (S_incoh(atom, |q|_center))`.
    2.  **Absolute Scale Factor Determination:**
        *   Determine the absolute scale factor `Scale_Abs` by fitting the *uncorrected* merged Bragg intensities to Wilson statistics of `F_calc_sq` in resolution shells (Martin & Withers 1982), thereby side-stepping partiality uncertainties.
    3.  **Apply Absolute Scale and Subtract Incoherent Scattering:**
        *   For each voxel in `VoxelData_relative` (solid-angle formula cross-validated against DIALS at θ = 0°, 30°, 60°):
            *   `I_abs_diffuse_voxel = VoxelData_relative.I_merged_relative(voxel) * Scale_Abs - I_incoherent_theoretical_voxel`.
            *   `Sigma_abs_diffuse_voxel = VoxelData_relative.Sigma_merged_relative(voxel) * Scale_Abs` (simplification, assumes `Scale_Abs` and `I_incoherent_theoretical_voxel` have negligible uncertainty relative to `Sigma_merged_relative`). This simplification must be documented.
*   **Output:**
    *   `VoxelData_absolute`: The final 3D diffuse scattering map on an absolute scale, with incoherent scattering removed. Structure similar to `VoxelData_relative` but with `I_abs_diffuse` and `Sigma_abs_diffuse`.
    *   `ScalingModel_final_list`: The per-still/group scaling models from `ScalingModel_refined_list` with the `Scale_Abs` factor incorporated into their overall scale components, and the `I_incoherent_theoretical` (as a function of `|q|`) incorporated into their additive offset components.
*   **Testing (Module 4.S.1):**
    *   **Input:** Sample `VoxelData_relative`, known `unitCellInventory`, `Crystal_avg_ref`, and a set of correctly prepared (scaled and partiality-corrected) Bragg intensities with known absolute scale.
    *   **Execution:** Call the absolute scaling and incoherent subtraction function.
    *   **Verification:** Assert that the calculated `Scale_Abs` is correct. Assert that `I_abs_diffuse` in `VoxelData_absolute` matches expected values after scaling and subtraction of incoherent scattering (calculated using full tabulated cross-sections).
