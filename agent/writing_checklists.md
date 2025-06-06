**Guideline: Drafting Implementation Checklists**

**1. Purpose of this Guideline**

This document provides a standard template and conventions for creating detailed implementation checklists. These checklists are intended to guide an AI agent (or a human developer) through the systematic implementation of project phases and modules, ensuring all planned tasks, context priming, and progress tracking are consistently handled.

**2. Checklist Structure**

Each checklist should adhere to the following overall structure:

```markdown
**Agent Task: [Brief, High-Level Task Title, e.g., Implement Phase X of Y Plan]**

**Overall Goal for Agent ([Phase/Task Name]):** [One or two sentences describing the primary objective of this specific checklist/phase.]

<checklist instructions>
**Checklist Usage Instructions for Agent:**

1.  **Copy this entire checklist into your working memory or a dedicated scratchpad area.**
2.  **Context Priming:** Before starting a new major section (e.g., Phase, Module), carefully read all "Context Priming" items for that section.
3.  **Sequential Execution:** Address checklist items in the order presented, unless an item explicitly states it can be done in parallel or depends on a later item being drafted first (e.g., an IDL definition).
4.  **Update State:** As you work on an item, change its state field:
    *   `[ ] Open` -> `[P] In Progress` when you start.
    *   `[P] In Progress` -> `[D] Done` when completed successfully.
    *   `[P] In Progress` -> `[B] Blocked` if you encounter a blocker. Add a note explaining the blocker in the "Details/Notes/Path" column.
5.  **Record Details (in the "Details/Notes/Path" column):**
    *   If a step requires creating or modifying a file, add the **full relative path** to that file (e.g., `src/package/module.py`).
    *   If a significant design decision or clarification is made during the task, note it briefly.
    *   If a task is broken down into further sub-tasks not originally listed, add them as indented items with their own Item ID and State.
    *   For "IDL Definition/Review" tasks, summarize the key interface aspects (Inputs, Outputs, Behavior, Errors) or reference the document where these are detailed.
6.  **Iterative Review:** Periodically re-read completed sections of the checklist and your notes to ensure continued alignment with the overall plan and previous decisions.
7.  **Save State:** If work needs to be paused, ensure this checklist with its current progress and notes is saved so work can be resumed effectively.
</checklist instructions>

---

**[Phase/Major Section Title, e.g., Phase X: Section Title]**

| Item ID | Task Description                                     | State | Details/Notes/Path                                     |
| :------ | :--------------------------------------------------- | :---- | :----------------------------------------------------- |
| **X.A** | **Context Priming ([Phase/Section Name])**           | `[ ]` |                                                        |
| X.A.1   | [Specific document/section to review for context]    | `[ ]` | [Optional: Note on what to focus on during review]     |
| X.A.2   | [Another document/section or concept to understand]  | `[ ]` |                                                        |
| ...     |                                                      |       |                                                        |
| X.A.N   | Understand Goal: [Restate goal for this specific section] | `[ ]` |                                                        |
|         |                                                      |       |                                                        |
| **X.B** | **[First Major Group of Tasks/Module within Phase]** | `[ ]` |                                                        |
| **X.B.idl**| **Define/Review Conceptual IDL for [Component/Module]** | `[ ]` | **Purpose:** [Brief purpose of this IDL step]. <br>Input: [Key inputs]. <br>Output: [Key outputs]. <br>Behavior: [Core behavior summary]. <br>Errors: [Key error conditions]. <br>[Optional: Path to where IDL thoughts are captured, e.g., comments in the target Python file, or a link to a formal IDL doc if it exists.] |
| X.B.1   | [Specific implementation task derived from plan]     | `[ ]` | [Note expected file path, key decisions, or dependencies] |
| X.B.1.a |   [Sub-task if needed]                               | `[ ]` |                                                        |
| X.B.1.b |   [Another sub-task]                                 | `[ ]` |                                                        |
| X.B.2   | [Another specific implementation task]               | `[ ]` |                                                        |
| ...     |                                                      |       |                                                        |
| X.B.T   | **Unit/Integration Tests for [Component/Module X.B]**| `[ ]` | Path: `tests/path/to/test_module_xb.py`                |
| X.B.T.1 |   Test Case: `test_scenario_one`                     | `[ ]` |                                                        |
|         |     - Setup: [Brief setup description]               | `[ ]` | [Note any specific test data files used]               |
|         |     - Execution: [How the component is called]       | `[ ]` |                                                        |
|         |     - Verification: [Key assertions]                 | `[ ]` |                                                        |
| ...     |                                                      |       |                                                        |
|         |                                                      |       |                                                        |
| **X.C** | **[Second Major Group of Tasks/Module within Phase]**| `[ ]` |                                                        |
| ...     | *(Repeat structure as for X.B)*                      |       |                                                        |
|         |                                                      |       |                                                        |
| **X.Z** | **[Phase/Section] Review & Next Steps**              | `[ ]` |                                                        |
| X.Z.1   | Self-Review: All [Phase/Section] items addressed?    | `[ ]` | [Confirm IDLs defined/reviewed, code implemented, tests written] |
| X.Z.2   | Context Refresh: Re-read [relevant plan sections for next phase]. | `[ ]` |                                                        |
| X.Z.3   | Decision: Proceed to [Next Phase/Section] Checklist. | `[ ]` |                                                        |

---
```

**3. Key Elements of a Checklist Item:**

*   **Item ID:**
    *   Hierarchical (e.g., `X.A.1`, `X.B.1.a`).
    *   `X` represents the Phase number or a major section letter.
    *   Second letter (`A`, `B`, `C`) represents a major group of tasks or a module within that phase.
    *   Numbers (`1`, `2`, `3`) represent specific tasks.
    *   Lowercase letters (`a`, `b`, `c`) represent sub-tasks.
    *   Use `**.idl**` suffix for items specifically about defining/reviewing an Interface Definition (conceptual or formal).
    *   Use `**.T**` suffix for items grouping test cases for a module.

*   **Task Description:**
    *   Clear, concise, and actionable.
    *   Start with a verb (e.g., "Implement...", "Define...", "Review...", "Test...").
    *   Derived directly from the project plan (e.g., `plan.md`).
    *   If it's a high-level task, break it down into sub-tasks.
    *   For "IDL Definition/Review" tasks, use the "Details/Notes/Path" column to sketch out:
        *   A brief **Purpose** statement for the interface.
        *   Key **Input(s)**.
        *   Key **Output(s)**.
        *   A summary of its core **Behavior**.
        *   Anticipated **Error(s)** or exceptional conditions.
        *   Optionally, the intended file path for the Python implementation of this interface.

*   **State:**
    *   A single field indicating current progress.
    *   Use predefined state markers:
        *   `[ ]` (Open / To Do)
        *   `[P]` (In Progress)
        *   `[D]` (Done / Completed)
        *   `[B]` (Blocked)

*   **Details/Notes/Path:**
    *   **Crucial for context and tracking.**
    *   **File Paths:** For tasks involving file creation/modification, list the full relative path to the primary file(s) involved.
    *   **Decisions:** Briefly note any significant design choices, clarifications, or assumptions made while performing the task if they deviate from or elaborate on the plan.
    *   **Blockers:** If State is `[B]`, explain the blocker here.
    *   **References:** Links to specific sections of other documents if helpful.
    *   For IDL tasks, this column should contain the sketch of the interface as described above.

**4. Content Guidelines:**

*   **Granularity:** Aim for tasks that are manageable units of work. A single task shouldn't be overly broad (e.g., "Implement entire module"). Break down larger tasks from the plan into smaller checklist items.
*   **IDL-First:** For any new component, class, or significant function, include an "IDL Definition/Review" item *before* its corresponding Python implementation item(s). This reinforces thinking about the interface first.
*   **Testing:** Include specific groups of test cases for each implemented module or significant piece of functionality. Test descriptions should cover setup, execution, and verification.
*   **Context Priming:** Each major section (Phase or significant Module group) should start with "Context Priming" tasks to ensure the agent re-orients itself with relevant plans and documents.
*   **Review and Next Steps:** Each major section should end with a review item and a clear pointer to the next steps or checklist.
*   **Consistency with `plan.md`:** Task descriptions should closely mirror the language and intent of the corresponding sections in `plan.md` or other planning documents.

**5. Formatting:**

*   Use Markdown tables for the main checklist structure.
*   Use bolding for section titles (e.g., `**X.A Context Priming...**`) and for emphasizing the "IDL Definition/Review" task type.
*   Use fixed-width font for `[ ]`, `[P]`, `[D]`, `[B]` states for visual clarity.
*   Use consistent indentation for sub-tasks if not using the table format for them (though the table format with hierarchical IDs is preferred).
