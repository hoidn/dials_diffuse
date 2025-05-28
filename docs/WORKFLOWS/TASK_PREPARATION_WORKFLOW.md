# Tech Lead Workflow: Task Preparation & Instruction Generation

**Purpose:** This guide outlines a standard process for a Tech Lead (TL) or senior developer to prepare a development task based on requirements and IDL specifications. The key output of this process is a detailed instruction document for the implementing developer.

**Goal:** To ensure tasks are well-defined, architecturally sound, and have a clear implementation and testing plan before coding begins, facilitating a smooth handoff and successful implementation.

---

**Phase 0: Preparation & Context Gathering**

*   **Goal:** Thoroughly understand the task requirements, the primary component's contract (IDL), and the context of its dependencies and interactions.
*   **Actions:**
    1.  **Define Task:** Clearly understand the overall goal (e.g., from an issue tracker, user story, feature request). Identify the scope and desired outcome.
    2.  **Locate Primary IDL File(s):** Find the main IDL file(s) (e.g., `*_IDL.md`) for the component(s) being implemented or modified. If no IDL exists for modifications, consider creating/updating one first as per `01_IDL_GUIDELINES.md`.
    3.  **Analyze IDL Contract:**
        *   Read the primary IDL(s) carefully.
        *   Identify: Interface/Module Purpose, Method/Function Signatures, Preconditions, Postconditions, Documented Behavior, Error Conditions, Expected Data Structures. Note any ambiguities.
    4.  **Identify & Review Dependencies:**
        *   List all declared dependencies (e.g., `@depends_on`, `@depends_on_resource`) from the primary IDL(s).
        *   Locate and **review the IDLs** for any internal components listed as dependencies. Understand *their* contracts.
    5.  **Study Relevant External/Internal APIs & Libraries:**
        *   Based on the task goal and dependencies, identify relevant external libraries, third-party tools, internal APIs, or system primitives that will be used.
        *   **Action:** **Thoroughly consult** the relevant documentation:
            *   `../LIBRARY_INTEGRATION/` for guides on key project libraries.
            *   `../ARCHITECTURE/types.md` or component-specific IDLs for internal APIs/protocols/types.
            *   Official external documentation for third-party libraries/tools.
        *   **Goal:** Fully understand the specific functions/methods/endpoints to call, required parameters, expected data formats, and error handling of these dependencies.
    6.  **Review Related Project Docs:** Check relevant ADRs (`../ARCHITECTURE/adr/`), architectural patterns, and project rules (`02_IMPLEMENTATION_RULES.md`, `03_PROJECT_RULES.md`).
    7.  **Formulate Testing Strategy:** Based on dependencies and project guidelines (see `02_IMPLEMENTATION_RULES.md`), decide the primary testing approach (e.g., prioritize integration tests, identify necessary unit tests, determine key boundaries for mocking).
    8.  **Update Working Memory:** Record the task, key findings, reviewed documents, identified risks/ambiguities, and initial testing strategy in your working memory log (see `../TEMPLATES/WORKING_MEMORY_LOG_TEMPLATE.md`).

**Phase 1: Stubbing & Plan Generation**

*   **Goal:** Create the necessary code/test skeletons and generate a detailed, actionable instruction document for the implementing developer.
*   **Actions:**
    1.  **Stub Skeleton Code:** Create basic code file(s) and class/function structures matching the IDL. Implement exact signatures with type hints. Copy/adapt IDL documentation into docstrings/comments. Add placeholder bodies (`pass` or `raise NotImplementedError`).
    2.  **Stub Tests & Outline Strategy:** Create empty test methods corresponding to key aspects identified in Phase 0 (success paths, error conditions, edge cases). Briefly outline the overall testing strategy in comments.
    3.  **Compile Detailed Implementation Plan for Developer:**
        *   Based on the IDL behavior, dependency knowledge (Phase 0, Step 5), and architectural constraints, outline specific implementation steps.
        *   Specify algorithms, data structures (e.g., from `../ARCHITECTURE/types.md` or new ones to define).
        *   Detail *exactly* how to interact with dependencies (methods, parameters, handling results/errors).
        *   Specify precise error handling logic.
        *   **Define Host Language/DSL Boundary (If Applicable):** Clearly specify which parts of the logic involve complex data preparation (to be done in the host language) and which parts involve DSL/script execution. Detail how data prepared in the host language should be bound to the DSL environment.
    4.  **Compile Detailed Testing Plan for Developer:**
        *   For each stubbed test function:
            *   Specify required test framework fixtures.
            *   Detail necessary mock configurations (e.g., `mock_dependency.method.return_value = ...`, `mock_dependency.method.side_effect = ...`).
            *   List the *exact* assertions needed.
    5.  **Add Execution & Debugging Guidance for Developer:**
        *   Provide specific commands to run relevant tests.
        *   Include debugging tips relevant to the task.
        *   Clarify when and how the developer should seek help.
    6.  **Define Definition of Done:** Create a checklist outlining criteria for task completion (e.g., code implemented per plan, all specified tests passing, linting/formatting clean, self-review done, IDL updated if contract changed).
    7.  **Assemble Instruction Document:** Collate all the above details into the **Task Instruction Template** (see `../TEMPLATES/TASK_INSTRUCTION_TEMPLATE.md`). Ensure clarity, precision, and completeness.

**Phase 2: Handoff & Follow-up**

*   **Goal:** Assign the task with clear instructions and ensure necessary code stubs are available.
*   **Actions:**
    1.  **Finalize & Review Instructions:** Read through the completed instruction template.
    2.  **Commit Stubs:** Commit the stubbed code and test files created in Phase 1.
    3.  **Assign Task:** Assign the task to the developer via the issue tracker, linking to the detailed instruction document and relevant commit/branch.
    4.  **Plan Code Review:** Mentally note or schedule time for reviewing the developer's implementation upon completion.
