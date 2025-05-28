# Refactoring Guide

**1. Purpose**

This guide provides conventions and a recommended workflow for refactoring code within this project. Refactoring is the process of restructuring existing computer code – changing the factoring – without changing its external behavior. It aims to improve non-functional attributes like readability, maintainability, simplicity, and adherence to design principles (like Single Responsibility Principle - SRP).

**Related Documents:**
*   `02_IMPLEMENTATION_RULES.md` (Coding standards)
*   `03_PROJECT_RULES.md` (Module length guideline, Version control workflow)
*   `TEMPLATES/WORKING_MEMORY_LOG_TEMPLATE.md` (Developer working memory log)

**2. When to Refactor**

Consider refactoring when you encounter:

*   **Code Smells:** Duplicated code, long methods/classes, complex conditional logic, excessive parameters, tight coupling, etc.
*   **Module Length:** A module significantly exceeds the guideline set in `03_PROJECT_RULES.md` (e.g., > `[e.g., 300-500]` LoC).
*   **Complexity:** A component becomes difficult to understand, test, or modify.
*   **Poor Cohesion:** A module or class handles too many unrelated responsibilities.
*   **New Requirements:** Adapting existing code for new features reveals structural weaknesses.
*   **Technical Debt:** Addressing known shortcuts or suboptimal designs.

**3. General Principles**

*   **Small, Incremental Steps:** Break down large refactorings into smaller, manageable changes.
*   **Test Frequently:** Run tests after each small step to ensure behavior is preserved.
*   **Preserve External Behavior:** The primary goal is to improve internal structure *without* altering how the component interacts with others (its public contract/API, as defined in its IDL specification). If the contract *must* change, it's technically more than just refactoring and requires careful consideration of dependents and updates to the IDL specification.
*   **Improve Structure:** Focus on enhancing clarity, reducing complexity, and improving modularity (e.g., better adherence to SRP).
*   **Use Version Control:** Commit frequently with clear messages describing each refactoring step.

**4. Formalized Refactoring Decision Workflow**

This section outlines a structured approach for identifying refactoring needs, analyzing candidates, choosing a strategy, and verifying the results.

**Phase 1: Identification & Initial Assessment**

*   **Input:**
    *   Code base or component under review.
    *   Project quality metrics (if available).
    *   Developer observations/pain points.
*   **Actions:**
    *   Scan for code smells.
    *   Review module length against guidelines in `03_PROJECT_RULES.md`.
    *   Identify components with high change frequency or bug density.
    *   Note areas where developers frequently express confusion or frustration.
    *   Check for violations of project architecture or implementation rules.
    *   Run automated checks (e.g., line count, cyclomatic complexity) to identify overly long or complex components.
*   **Output:**
    *   List of refactoring candidates with brief descriptions of issues.
    *   Initial severity assessment (High/Medium/Low) based on impact on development velocity, risk of bugs, maintenance burden, and violation of core architectural principles.

**Phase 2: Detailed Analysis**

*   **Input:**
    *   Refactoring candidates from Phase 1.
    *   Relevant documentation (IDL files, architecture diagrams).
    *   Test coverage reports.
*   **Actions:**
    *   For each high-priority candidate:
        *   Analyze dependencies (what components depend on this code?).
        *   Assess test coverage (is behavior well-tested?).
        *   Identify specific design problems (SRP violations, tight coupling, etc.).
        *   Analyze complexity metrics (e.g., cyclomatic complexity).
        *   Determine root causes.
        *   Estimate effort required to refactor.
        *   Evaluate risks (complexity, potential for regression).
*   **Output:**
    *   Detailed analysis document for each candidate, including: specific problems, dependencies, test coverage, effort estimate, risk assessment.

**Phase 3: Strategy Selection**

*   **Input:**
    *   Detailed analysis from Phase 2.
    *   Project priorities and constraints.
    *   Available developer resources.
*   **Actions:**
    *   For each refactoring candidate, select an appropriate strategy from common refactoring techniques (e.g., Extract Method/Class, Move Method/Field, Replace Conditional with Polymorphism, Introduce Parameter Object, Rename, etc.).
*   **Output:**
    *   Refactoring plan for each candidate, including: selected strategy, specific steps, acceptance criteria.

**Phase 4: Planning & Preparation**

*   **Input:**
    *   Refactoring plans from Phase 3.
    *   Project schedule and priorities.
    *   Test infrastructure status.
*   **Actions:**
    *   Break down large refactorings into smaller, incremental steps.
    *   Identify or create tests that verify the current behavior. **Ensure test coverage is adequate before starting.**
    *   Schedule refactoring work.
    *   Update your working memory log (e.g., `TEMPLATES/WORKING_MEMORY_LOG_TEMPLATE.md`) with the refactoring task.
*   **Output:**
    *   Detailed, step-by-step refactoring plan with: specific files/sections to modify, order of operations, test strategy for each step, rollback plan, documentation updates needed.

**Phase 5: Execution (Iterative)**
*(This phase aligns with the "Recommended Refactoring Execution Workflow" in Section 6)*

*   **Actions:**
    *   Implement the refactoring in small, incremental steps.
    *   Run tests frequently after each step.
    *   Commit changes regularly with clear messages.
    *   Adapt tests as the internal structure changes, ensuring they still verify the external contract.

**Phase 6: Verification & Cleanup**

*   **Input:**
    *   Completed refactoring changes.
    *   Test results.
    *   Code review feedback.
*   **Actions:**
    *   Verify all tests pass (existing and new).
    *   Conduct code review.
    *   Check for any unintended side effects or regressions.
    *   Update documentation (IDL specifications, architecture diagrams if impacted) to reflect the new structure.
    *   Remove any dead code or unused imports.
    *   Run linters and formatters.
    *   Update your working memory log with the completed refactoring.
*   **Output:**
    *   Verified, clean, refactored code.
    *   Updated documentation.
    *   Lessons learned.

**6. Recommended Refactoring Execution Workflow**

This workflow details the execution steps once a refactoring is planned:

1.  **Identify & Plan (Covered in Phases 1-4 above):**
    *   Clearly define scope and target structure.
    *   Log task in your working memory.

2.  **Ensure Test Coverage (Before Refactoring):**
    *   Review existing tests for the code being refactored.
    *   **Crucial:** If coverage is insufficient, **write tests first** to capture the current behavior.

3.  **Extract Mechanically (Example: Extracting a Class/Function):**
    *   Create the new function, class, or module.
    *   Carefully move the identified code block(s) to the new location.
    *   Ensure necessary imports are added to the new location.

4.  **Delegate / Integrate:**
    *   Modify the original code location to *call* or *use* the newly extracted code/component.
    *   Pass necessary data or dependencies to the new code.
    *   Ensure imports are updated in the original location.

5.  **Adapt Tests (After Refactoring):**
    *   Run existing tests; expect some to fail.
    *   **Modify Original Tests:**
        *   Update tests that relied on the *internal implementation* that was moved.
        *   Mock the *newly extracted dependency* (the function/class you created).
        *   Change assertions to verify that the original code now correctly *delegates* the call.
        *   Remove assertions that tested internal logic now moved.
    *   **Add New Tests:**
        *   Write new, focused tests for the *extracted component*.
    *   Run all relevant tests again until they pass.

6.  **Clean Up:**
    *   Remove any dead code from the original location.
    *   Run linters and formatters.
    *   Review changes for clarity and simplicity.

7.  **Document:**
    *   Update your working memory log.
    *   If public interfaces or significant structures changed, update relevant IDL specifications or architecture diagrams.

**7. Common Pitfalls & How to Avoid Them**

*   **Breaking Behavior:**
    *   **Avoidance:** Good test coverage *before* starting. Test frequently.
*   **Brittle Tests:** Tests failing due to internal restructuring.
    *   **Avoidance:** Test the component's *contract* and *behavior*, not its specific internal implementation. Mock primarily at boundaries.
*   **Incorrect Mocking:** `patch` failing or mocks not behaving as expected.
    *   **Avoidance:** Understand how to use your mocking framework (e.g., `patch` where it's *looked up*). Verify mock calls.
*   **Scope Issues in Tests:** Mock instances or fixtures not available.
    *   **Avoidance:** Understand test fixture scopes.
*   **Trying to Do Too Much:**
    *   **Avoidance:** Small, incremental steps. Commit after each logical step passes tests.
*   **Forgetting Cleanup:** Leaving unused imports, variables, or methods.
    *   **Avoidance:** Explicitly review for dead code. Use linters.

**8. Example (Conceptual: Extracting a Helper Function)**

1.  **Identify:** A complex calculation is repeated within a method `process_data`.
2.  **Test (Before):** Ensure `test_process_data` covers cases involving this calculation.
3.  **Extract:** Create a new private method `_perform_calculation(input)`. Move logic there.
4.  **Delegate:** Replace logic in `process_data` with a call to `self._perform_calculation(input)`.
5.  **Test (After):** `test_process_data` should ideally still pass. Add `test__perform_calculation`.
6.  **Clean Up:** Format/lint.
7.  **Document:** Update working memory log.

By following this guide, we can iteratively improve the codebase's quality while minimizing disruption and maintaining correctness.
