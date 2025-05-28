# Guide: Keeping Documentation Consistent

**1. Purpose**

This guide outlines a standard process for reviewing and updating project documentation to ensure it accurately reflects the current state of the codebase, architecture, and development practices. Consistent and up-to-date documentation is crucial for onboarding new developers, maintaining architectural integrity, and reducing confusion.

**2. When to Use This Process**

This process should be triggered:

*   **After Significant Technical Changes:** Following the merge of code that implements a major feature, refactoring, architectural decision (ADR), or change in core dependencies.
*   **Before Starting Major New Work:** As part of the preparation phase for a new feature that builds upon existing components, verifying the documentation for those components is accurate.
*   **Periodically:** As part of scheduled maintenance to catch drift between documentation and implementation over time.

**3. Goal**

To identify and rectify inconsistencies, inaccuracies, or outdated information within the project's documentation (typically the `docs/` directory and potentially IDL files).

**4. The Process**

**Phase 1: Define Scope & Identify Affected Documents**

1.  **Identify the Trigger/Scope:** Clearly define the technical change or area of focus that necessitates the documentation review.
    *   *Example Trigger:* "ADR-XYZ (New Authentication System) has been implemented."
    *   *Example Scope:* "Review documentation related to user authentication, security, and relevant component interfaces."

2.  **List Potentially Affected Documents:** Brainstorm and list all documentation files that *might* be impacted by the change or are relevant to the area of focus.
    *   **Core Guides:** Always check `00_START_HERE.md`, `03_PROJECT_RULES.md`, `02_IMPLEMENTATION_RULES.md`, `01_IDL_GUIDELINES.md`.
    *   **Project Plan/Roadmap (if applicable):** Check any high-level planning documents if the technical change impacts scope, timelines, or dependencies.
    *   **Related ADRs:** Review the ADR itself and any ADRs it supersedes or relates to (see `docs/ARCHITECTURE/adr/`).
    *   **Component Docs & IDL Files:** Identify components directly affected. Check their IDL files (e.g., `src/component_x/component_x_IDL.md`), READMEs, or other specific documentation.
    *   **Shared Types/Contracts:** Check `docs/ARCHITECTURE/types.md` if the change impacts shared data structures or protocols.
    *   **Architectural Patterns:** Check `docs/ARCHITECTURE/patterns/` (if such a directory exists) if the change implements or modifies a core pattern.
    *   **Search:** Use project-wide search tools for keywords related to the change across the `docs/` directory (and `src/` for IDL files) to find less obvious references.
    *   *Example Output for New Auth System:* `00_START_HERE.md`, `02_IMPLEMENTATION_RULES.md` (security section), `src/auth_service/auth_service_IDL.md`, `src/user_component/user_component_IDL.md`, `docs/ARCHITECTURE/adr/ADR-XYZ.md`.

**Phase 2: Review and Analyze**

1.  **Read Through Identified Documents:** Carefully read each document identified in Phase 1.
2.  **Compare Against Reality:** Compare the documentation against:
    *   The **specific technical changes** that triggered the review.
    *   The **current codebase structure**.
    *   The **current behavior** of the code (if known or testable).
    *   Other **related documentation** (check for contradictions).
3.  **Identify Gaps and Inconsistencies:** Look for:
    *   **Outdated Information:** Descriptions, code examples, diagrams, or procedures that no longer match the implementation.
    *   **Contradictions:** Information that conflicts with the implemented changes or other documentation.
    *   **Inconsistent Terminology:** Using different names for the same concept across documents.
    *   **Broken Links/References:** Cross-references pointing to non-existent files or sections.
    *   **Missing Information:** Failure to document new components, features, patterns, or conventions.
    *   **Ambiguity:** Sections that are unclear or open to multiple interpretations.
    *   *Example Gaps Found:* `02_IMPLEMENTATION_RULES.md` described old auth method; `00_START_HERE.md` pointed to wrong component for user data.

**Phase 3: Draft and Apply Updates**

1.  **Draft Targeted Edits:** For each identified gap, draft the specific changes needed.
    *   Focus on correcting inaccuracies or adding missing information.
    *   Ensure changes align with the project's documentation style.
    *   Update code examples, diagrams, and cross-references.
    *   Reference the trigger (e.g., "Updated to reflect New Authentication System per ADR-XYZ").
2.  **Apply Changes:** Edit the documentation files.
3.  **Self-Review:** Read through your changes. Are they accurate, clear, and easy to understand? Do they introduce new inconsistencies?

**Phase 4: Review and Commit**

1.  **Peer Review (Recommended):** If possible, have another team member review substantial documentation changes for clarity and accuracy.
2.  **Commit Changes:** Use clear, specific commit messages explaining the purpose of the documentation update. Reference the triggering change if applicable.
    *   *Example Commit Message:* `docs: Align auth guides with new system per ADR-XYZ`
    *   *Example Commit Message:* `docs: Clarify data model usage in component_x_IDL.md`

**5. Key Considerations**

*   **Scope:** Keep the update focused on the triggering change or area under review.
*   **Single Source of Truth:** Ensure information is updated in the *authoritative* source document. Avoid duplicating detailed explanations; use cross-references.
*   **Clarity:** Write for someone unfamiliar with the specific change or component. Define terms or link to definitions.
*   **Consistency:** Ensure terminology, formatting, and style are consistent with surrounding documentation.

By following this process, we can maintain accurate, consistent, and helpful documentation that reflects the ongoing evolution of the project.
