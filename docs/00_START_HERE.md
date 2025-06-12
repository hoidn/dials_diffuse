# Developer Orientation: Getting Started with DiffusePipe

**Welcome!** This document is your starting point for understanding how we build software in this project. Our approach emphasizes clear specifications and consistent implementation patterns. Please read this guide carefully before diving into the code.


**1. Core Philosophy: Interface Definition Language (IDL) as the Specification**

The cornerstone of our development process is the use of **Interface Definition Language (IDL)** files (e.g., `*_IDL.md` or similar, defining the contract for components).

*   **The IDL is the Contract:** Each IDL file defines a strict **contract** for a specific component (module or interface). It specifies *what* the component must do, its public interface, its expected behavior, its dependencies, and its error conditions. Implementations must not only match the public interface but also adhere to the behavioral descriptions, preconditions, postconditions, and specified data structures (like required dictionary keys) defined in the IDL.
*   **Specification, Not Implementation:** The IDL focuses on the **behavioral specification** and separates it from the *how* (the specific implementation details).
*   **Source of Truth:** When implementing a component, its corresponding IDL file is the authoritative source for its requirements.
*   **Bidirectional Goal:** We use IDLs both to *specify* new components before coding (IDL-to-Code) and to *document* the essential contract of existing components by abstracting away implementation details (Code-to-IDL).

> **Further Reading:** For the detailed syntax and rules for creating/reading IDL specifications, refer to `01_IDL_GUIDELINES.md`.

**2. Understanding the IDL Structure (`*_IDL.md` files)**

When you open an IDL file (e.g., `src/component_x/component_x_IDL.md`), you'll typically find:

*   **Module/Component Path:** Defines the logical grouping, corresponding to the code module path (e.g., `src.component.path`).
*   **Dependency Declarations (e.g., `@depends_on(...)`):** Declares dependencies on *other IDL-defined interfaces/modules* or *abstract external resources* (like `FileSystem`, `Database`, `ExternalAPI`). These signal required interactions.
*   **Interface/Class Definition:** Defines the contract for a specific class or a collection of related functions. The name typically matches the code class/module name.
*   **Method/Function Definitions (`returnType methodName(...)`)**:
    *   **Signature:** Specifies the method name, parameter names, parameter types (`string`, `int`, `list<Type>`, `dict<Key,Val>`, `optional Type`, `union<T1, T2>`, `object` for other interfaces), and the return type.
    *   **Documentation Block (CRITICAL - This IS the Spec!):**
        *   **`Preconditions:`**: What must be true *before* calling the method.
        *   **`Postconditions:`**: What is guaranteed *after* the method succeeds (return values, state changes).
        *   **`Behavior:`**: A description of the essential logic/algorithm the method performs and how it interacts with dependencies.
        *   **`Expected Data Format: { ... }`**: If a parameter or return type is a complex dictionary/object, its structure is defined here (e.g., using JSON-like notation or by referencing a defined `struct`).
        *   **`@raises_error(condition="ErrorCode", ...)` (or similar annotation):** Defines specific, contractual error conditions the method might signal.
    *   **`Invariants:`** (Optional): Properties of the component's state that should always hold true between method calls.
*   **Custom Data Structure Definitions (e.g., `struct StructName { ... }`):** Defines reusable, complex data structures. These might be defined within the IDL file itself or in a central types definition file (e.g., `ARCHITECTURE/types.md`) for globally shared types.
*   **Component Interactions Section (Optional):**
    *   Provides supplementary documentation about how the defined module or interface interacts with its dependencies for common or complex workflows.
    *   Often includes **sequence diagrams (e.g., using Mermaid syntax)** and textual explanations to clarify call chains and data flow between components.

**3. The Development Workflow: Implementing from IDL Specifications**

When assigned to implement or modify a component specified by an IDL (or tackling a new feature):

**Phase 0: Preparation & Understanding**

1.  **Define Task:** Clearly understand the goal (e.g., from an issue tracker, user story).
2.  **Locate/Review IDL:** Find the relevant IDL file(s). If modifying existing code without a complete IDL, consider generating/updating the IDL first (Code-to-IDL).
3.  **Understand Contract & Interactions:** Thoroughly read the IDL: purpose, dependencies, function/method signatures, and especially the documentation blocks (`Preconditions`, `Postconditions`, `Behavior`, `Expected Data Format`, error conditions). Look for any "Component Interactions" section. Pay close attention to dependency declarations.
4.  **Review Rules:** Briefly refresh understanding of key guidelines in `02_IMPLEMENTATION_RULES.md` (especially Testing) and `03_PROJECT_RULES.md`.
5.  **Outline Testing Strategy:** Based on the IDL dependencies and testing guidelines, outline the primary testing approach.
6.  **Setup Working Memory:** Update your working memory log (e.g., `TEMPLATES/WORKING_MEMORY_LOG_TEMPLATE.md`) with your "Current Task/Focus", initial "Testing Strategy Outline", and initial "Next Steps".

**Phase 1: Implementation & Testing ("Main Step")**

7.  **Refine Testing Strategy & Identify Mock Targets:** Solidify the testing approach. If using mocks, explicitly identify patch targets.
8.  **Create/Modify Files:** Ensure code file/directory structure matches the IDL module path.
9.  **Implement Structure & Dependencies:** Create Python classes/functions. Implement constructors, injecting dependencies specified in the IDL.
10. **Implement Functions/Methods:** Define signatures exactly matching the IDL. Implement logic described in `Behavior`. Use defined data models (e.g., Pydantic) for complex data. Fulfill `Postconditions`. Respect `Preconditions`. Implement error handling for specified conditions.
11. **Write Tests (Unit and/or Integration):** Implement tests verifying the implementation against the *entire* IDL contract. Follow Arrange-Act-Assert. Use fixtures.
12. **Log Progress:** Update your working memory log.

**Phase 2: Finalization & Sanity Checks ("Cleanup Step")**

13. **Format Code:** Run the project's code formatter.
14. **Lint Code:** Run the project's linter and address issues.
15. **Run All Tests:** Execute the full test suite.
16. **Perform Sanity Checks:** Self-review. Check against `02_IMPLEMENTATION_RULES.md` and `03_PROJECT_RULES.md`. Update directory structure docs if needed. Verify code still matches the IDL contract. Check related configuration files (e.g., test setups).
17. **Finalize Working Memory:** Update with final status and thoughts.

---

**4. Visual Diagnostics & Testing**

*   **Visual Diagnostics Guide:** See `docs/VISUAL_DIAGNOSTICS_GUIDE.md` for comprehensive documentation on visual verification tools for Phase 2 data extraction.
*   **End-to-End Testing:** Use `scripts/dev_workflows/run_phase2_e2e_visual_check.py` for complete pipeline verification.
*   **Diagnostic Tools:** Use `scripts/visual_diagnostics/check_diffuse_extraction.py` for detailed analysis of extraction outputs.
*   **DIALS Debugging:** See `docs/06_DIALS_DEBUGGING_GUIDE.md` for troubleshooting DIALS integration issues.

---

**5. Key Coding Standards**

*   **Python Version:** `3.10+`
*   **Formatting:** PEP 8 compliant, enforced by `Black` & `Ruff` (run `make format` / `make lint` or equivalent).
*   **Type Hinting:** **Mandatory** for all signatures. Use standard `typing` types. Be specific.
*   **Docstrings:** **Mandatory** for all modules, classes, functions, methods. Use **Google Style** docstrings.
*   **Imports:** Use **absolute imports** from `src` (or your project's source root). Group imports correctly. Top-level only unless exceptional reason exists (document it).
*   **Naming:** PEP 8 (snake_case for functions/variables, CamelCase for classes). Use descriptive names.
*   **Logging:** Use the standard `logging` module. Configure logging early in entry-point scripts as detailed in `02_IMPLEMENTATION_RULES.md`.

> **Further Reading:** See `02_IMPLEMENTATION_RULES.md` and `03_PROJECT_RULES.md` for complete details.

**5. Important Patterns & Principles**

*   **Parse, Don't Validate (e.g., with Pydantic):**
    *   **Concept:** Instead of validating raw data (like dicts) inside your logic, parse it into a well-defined model (e.g., a Pydantic `BaseModel`) at the component boundary.
    *   **Practice:** Define models for complex data structures. Use model validation methods to parse inputs.
    *   **Reference:** See relevant section in `02_IMPLEMENTATION_RULES.md`.
*   **Dependency Injection:**
    *   **Concept:** Components receive their dependencies via their constructor (`__init__`). Avoid global state or direct instantiation of dependencies within methods.
    *   **Practice:** Identify dependencies from IDLs. Add parameters to `__init__`. Store references as instance attributes.
*   **Separation of Atomic Units & Composition Logic:**
    *   **Concept:** Define small, focused, reusable units of functionality ("atomic units" or "primitives"). Use a separate composition layer or language (e.g., a DSL, Python scripting) to orchestrate these units into complex workflows.
    *   **Practice:** Ensure atomic units only use explicitly passed parameters. Ensure the composition layer handles all sequencing, conditionals, and loops.
*   **Interaction with External Services/APIs (e.g., LLMs, Databases):**
    *   **Standard:** Use a dedicated manager or bridge class to encapsulate interactions with significant external services.
    *   **Practice:** This manager class handles connection details, API call formatting, and basic response parsing. Other components use this manager rather than interacting directly with the external service's raw API.
    *   **Reference:** See relevant section in `02_IMPLEMENTATION_RULES.md` and project-specific library integration guides (e.g., in `LIBRARY_INTEGRATION/`).
*   **Integration with DIALS via Python API:**
    *   **Standard:** Interaction with DIALS crystallography software for initial image processing (spot finding, indexing, integration) is handled by a Python orchestrator (`StillsPipelineOrchestrator`). This orchestrator first determines the data type (true still or sequence data) by inspecting the CBF image header (Module 1.S.0).
    *   For **true still images** (Angle_increment = 0.0°), it uses the `dials.stills_process` Python API via a dedicated adapter (`DIALSStillsProcessAdapter`).
    *   For **sequence data** (Angle_increment > 0.0°), it uses a sequential DIALS command-line workflow (import → find_spots → index → integrate) via a separate adapter (`DIALSSequenceProcessAdapter`).
    *   Both adapters produce consistent output objects (`ExperimentList`, `reflection_table`) for downstream processing.
    *   **Practice:** The Python orchestrator configures and invokes the appropriate DIALS adapter based on data type. These adapters manage the internal DIALS workflow. Subsequent Python modules (`DataExtractor`, `ConsistencyChecker`, etc.) process and analyze the Python objects (e.g., DIALS `ExperimentList`, `reflection_table`) and files produced by this DIALS processing stage.
    *   **Geometric Corrections:** The `DataExtractor` leverages the robust DIALS `Corrections` API for Lorentz-Polarization and Quantum Efficiency corrections, while implementing custom calculations only for Solid Angle and Air Attenuation corrections specific to diffuse pixels.
    *   **Configuration:** PHIL files (e.g., in `src/diffusepipe/config/` or specified by the user) provide standardized parameter settings for `dials.stills_process`.
*   **Host Language Orchestration with DSL/Script Evaluation (if applicable):**
    *   **Concept:** For projects with an embedded Domain-Specific Language (DSL) or scripting capability, use the host language (e.g., Python) for complex data preparation. The DSL/script then focuses on orchestration, referencing data prepared by the host language.
    *   **Practice:** Python code prepares data, creates an environment for the DSL, binds data to variables, and then calls the DSL evaluator.
    *   **Reference:** See relevant section in `02_IMPLEMENTATION_RULES.md`.

**6. Testing Strategy**

*   **Framework:** `pytest`.
*   **Focus:** **Strongly emphasize Integration and Functional/End-to-End tests** over isolated unit tests. Components should be tested together to verify they work correctly according to their IDL contracts in realistic scenarios.
*   **Real Components Over Mocks:** **Avoid mocks whenever possible.** Use actual component implementations in tests to ensure behavior matches production. The goal is to verify real interactions, not theoretical ones.
*   **When to Use Real Components:**
    *   For all internal project components
    *   For file system operations (when reasonable)
    *   For database operations (using test databases)
    *   For any component where behavior can be realistically simulated
*   **Limited Mock Usage:** Only mock external services when absolutely necessary:
    *   Third-party APIs with usage limits or authentication requirements
    *   Services requiring complex infrastructure that can't be containerized
    *   Components with non-deterministic behavior that can't be controlled
*   **Error Path Testing:** Explicitly test error handling and propagation with real components.
*   **Fixtures:** Use testing framework fixtures for setup of test environments and dependencies.
*   **Structure:** Follow the `Arrange-Act-Assert` pattern. Mirror the `src` directory structure in `tests`.

> **Further Reading:** See relevant section in `02_IMPLEMENTATION_RULES.md`.

**7. Project Navigation (Actual Structure)**

*   **`src/` (source root):** Main application source code.
    *   `src/diffusepipe/`: Primary package
        *   `__init__.py`
        *   `adapters/`: DIALS/DXTBX API wrappers
            *   `dials_stills_process_adapter.py` and `*_IDL.md`
            *   `dials_sequence_process_adapter.py` and `*_IDL.md`
            *   `dials_generate_mask_adapter.py` and `*_IDL.md`
            *   `dxtbx_io_adapter.py` and `*_IDL.md`
        *   `config/`: Configuration files (PHIL files)
            *   `find_spots.phil`
            *   `refine_detector.phil`
            *   `sequence_*_default.phil` files
        *   `crystallography/`: Crystal model processing and validation
            *   `q_consistency_checker.py` and `*_IDL.md`
            *   `still_processing_and_validation.py` and `*_IDL.md`
        *   `diagnostics/`: Diagnostic tools
            *   `q_calculator.py` and `*_IDL.md`
        *   `extraction/`: Data extraction components
            *   `data_extractor.py` and `*_IDL.md`
        *   `masking/`: Pixel and Bragg mask generation
            *   `pixel_mask_generator.py` and `*_IDL.md`
            *   `bragg_mask_generator.py` and `*_IDL.md`
        *   `orchestration/`: Pipeline coordination
            *   `pipeline_orchestrator_IDL.md`
            *   `stills_pipeline_orchestrator_IDL.md`
        *   `types/`: Data type definitions
            *   `types_IDL.py` and `types_IDL.md`
        *   `utils/`: Utility functions
            *   `cbf_utils.py`
    *   `src/scripts/`: Processing scripts
        *   `process_pipeline.sh`: Main processing script
*   **`scripts/`**: Development and diagnostic scripts
    *   `scripts/dev_workflows/run_phase2_e2e_visual_check.py`: End-to-end pipeline verification
    *   `scripts/visual_diagnostics/check_diffuse_extraction.py`: Diagnostic plot generation
*   **`tests/`**: Test suite (integration-focused)
    *   Mirrors `src/` structure for easy navigation
    *   `tests/data/`: Test data files
*   **`libdocs/`**: External library documentation
    *   `libdocs/dials/`: DIALS/CCTBX/DXTBX documentation
*   **Project Configuration:**
    *   `CLAUDE.md`: Instructions for Claude AI assistant
    *   `plan.md`: Master technical implementation specification
    *   `pyproject.toml`: Python project configuration
*   **`docs/`**: All project documentation.
    *   `docs/00_START_HERE.md`: This file (developer onboarding guide)
    *   `docs/01_IDL_GUIDELINES.md`: IDL structure and syntax guidelines
    *   `docs/02_IMPLEMENTATION_RULES.md`: Code implementation standards
    *   `docs/03_PROJECT_RULES.md`: Project workflow and organization
    *   `docs/04_REFACTORING_GUIDE.md`: Guidelines for code refactoring
    *   `docs/05_DOCUMENTATION_GUIDE.md`: Documentation standards
    *   `docs/06_DIALS_DEBUGGING_GUIDE.md`: DIALS integration troubleshooting
    *   `docs/VISUAL_DIAGNOSTICS_GUIDE.md`: Visual verification tools for Phase 2
    *   `docs/LESSONS_LEARNED.md`: Project development insights
    *   `docs/ARCHITECTURE/`: Architecture documentation
        *   `adr/`: Architecture Decision Records
        *   `types.md`: Shared data structure definitions
    *   `docs/LIBRARY_INTEGRATION/`: External library integration guides
    *   `docs/TEMPLATES/`: Document templates
    *   `docs/WORKFLOWS/`: Workflow documentation
*   **`README.md`**: Top-level project overview.

**8. Development Workflow & Recommended Practices**

*   **Follow the IDL Specification:** Adhere strictly to the IDL for the component.
*   **Plan Your Tests:** Consider testing strategy *before* implementation.
*   **Use Working Memory:** Maintain a log of your progress (see `TEMPLATES/WORKING_MEMORY_LOG_TEMPLATE.md`).
*   **Be Aware of Existing Code & Configuration:** Consider impacts on related code and config files.
*   **Test Driven (where practical):** Write tests to verify against the IDL contract.
*   **Commit Often:** Small, logical commits with clear messages.
*   **Format and Lint:** Before committing.
*   **Ask Questions:** If unsure about requirements or design.

---

**9. External Dependencies**

*   **DIALS Crystallography Software:** The project relies heavily on DIALS for crystallographic data processing.
    *   Must be installed and available in the system PATH
    *   Commands used include: `dials.import`, `dials.find_spots`, `dials.index`, `dials.refine`, and `dials.generate_mask`
*   **PDB Files:** The project uses PDB files (e.g., `6o2h.pdb`) for crystallographic consistency checks.
    *   Used by the Python scripts to validate processing results against known structures
    *   Must be provided via the `--external_pdb` parameter to the processing script
*   **Python Dependencies:**
    *   `dxtbx`: DIALS Toolbox for image format reading
    *   `cctbx`: Computational Crystallography Toolbox
    *   `numpy`, `scipy`: Numerical processing
    *   `matplotlib`: Optional visualization
    *   `tqdm`: Progress bars

**10. Getting Started Checklist**

1.  [ ] Read this document (`00_START_HERE.md`).
2.  [ ] Read `01_IDL_GUIDELINES.md` to understand IDL guidelines.
3.  [ ] Read `02_IMPLEMENTATION_RULES.md` for detailed coding/testing rules.
4.  [ ] Read `03_PROJECT_RULES.md` for project structure and workflow.
5.  [ ] Review the main `README.md` and key architecture diagrams (e.g., in `docs/ARCHITECTURE/`).
6.  [ ] Set up your local development environment (`Python 3.10+`, DIALS, dependencies).
7.  [ ] Browse the `src/` directory and a few IDL files to see the structure.
8.  [ ] Try running the processing script with a test CBF file and PDB file.
9.  [ ] Ask questions!

Welcome aboard! By following these guidelines, we can build a robust, maintainable, and consistent system together.
