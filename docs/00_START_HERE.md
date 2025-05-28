# Developer Orientation: Getting Started with [YourProjectName]

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

**4. Key Coding Standards**

*   **Python Version:** `[Specify target Python version, e.g., 3.10+]`
*   **Formatting:** PEP 8 compliant, enforced by `[Formatter, e.g., Black]` & `[Linter, e.g., Ruff]` (run `make format` / `make lint` or equivalent).
*   **Type Hinting:** **Mandatory** for all signatures. Use standard `typing` types. Be specific.
*   **Docstrings:** **Mandatory** for all modules, classes, functions, methods. Use **`[Docstring Style, e.g., Google Style]`**.
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
*   **Integration with [ExternalToolName] via [Protocol/Library]:**
    *   **Standard:** Interaction with specific external tools (e.g., a code analysis tool, a specialized CLI) is handled via a dedicated "Bridge" component.
    *   **Practice:** The Bridge acts as a client to the external tool, potentially using a specific protocol or library. Application logic calls methods on the Bridge.
*   **Host Language Orchestration with DSL/Script Evaluation (if applicable):**
    *   **Concept:** For projects with an embedded Domain-Specific Language (DSL) or scripting capability, use the host language (e.g., Python) for complex data preparation. The DSL/script then focuses on orchestration, referencing data prepared by the host language.
    *   **Practice:** Python code prepares data, creates an environment for the DSL, binds data to variables, and then calls the DSL evaluator.
    *   **Reference:** See relevant section in `02_IMPLEMENTATION_RULES.md`.

**6. Testing Strategy**

*   **Framework:** `[Testing Framework, e.g., pytest]`.
*   **Focus:** Prioritize **Integration and Functional/End-to-End tests** over isolated unit tests. Verify components work together correctly according to their IDL contracts.
*   **Mocking:** **Minimize mocking.** Mock primarily at external boundaries (external APIs, services) or where strictly necessary. Prefer using real instances of internal components in integration tests.
*   **Error Path Testing:** Explicitly test error handling and propagation.
*   **Fixtures:** Use testing framework fixtures for setup.
*   **Structure:** Follow the `Arrange-Act-Assert` pattern. Mirror the `src` directory structure in `tests`.

> **Further Reading:** See relevant section in `02_IMPLEMENTATION_RULES.md`.

**7. Project Navigation (Example Structure)**

*   **`src/` (or your source root):** Main application source code.
    *   `src/component_a/`
    *   `src/component_b/`
    *   `src/system/` (for core system-wide utilities, models, errors)
*   **`tests/`**: Tests, mirroring the `src` structure.
*   **`docs/`**: All project documentation.
    *   `docs/01_IDL_GUIDELINES.md`
    *   `docs/ARCHITECTURE/types.md` (Shared data structure definitions)
    *   `docs/02_IMPLEMENTATION_RULES.md`
    *   `docs/03_PROJECT_RULES.md`
    *   `docs/examples/` (Example usage patterns)
    *   `src/**/[component_name]_IDL.md` (Specific interface definitions)
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

**9. Getting Started Checklist**

1.  [ ] Read this document (`00_START_HERE.md`).
2.  [ ] Read `01_IDL_GUIDELINES.md` to understand IDL guidelines.
3.  [ ] Read `02_IMPLEMENTATION_RULES.md` for detailed coding/testing rules.
4.  [ ] Read `03_PROJECT_RULES.md` for project structure and workflow.
5.  [ ] Review the main `README.md` and key architecture diagrams (e.g., in `docs/ARCHITECTURE/`).
6.  [ ] Set up your local development environment (`[Target Language/Platform]`, dependencies, pre-commit hooks).
7.  [ ] Browse the `src/` (or equivalent) directory and a few IDL files to see the structure.
8.  [ ] Try running the tests (`[test command, e.g., pytest tests/]`).
9.  [ ] Ask questions!

Welcome aboard! By following these guidelines, we can build a robust, maintainable, and consistent system together.
