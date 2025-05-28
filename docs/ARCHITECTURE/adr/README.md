# Architecture Decision Records (ADRs)

This directory stores Architecture Decision Records (ADRs) for the `[YourProjectName]` project.

## What is an ADR?

An ADR is a short document that captures a significant architectural decision made along with its context and consequences. ADRs are immutable once "Accepted"; if a decision is changed, a new ADR is created that supersedes the old one.

The goal of using ADRs is to:
*   Document important architectural choices and their rationale.
*   Provide context for future developers or team members.
*   Avoid re-litigating past decisions without new information.
*   Communicate architectural changes clearly.

## When to Create an ADR

Consider creating an ADR when:
*   Making a choice that has a significant impact on the system's architecture (e.g., structure, non-functional characteristics, dependencies, interfaces between components).
*   Introducing a new technology, library, or framework that is core to the system.
*   Choosing a specific pattern or approach over others for a critical part of the system.
*   Changing a previous architectural decision.
*   Deprecating a major feature, component, or architectural pattern.

If a decision is small, local to a single component, and easily reversible, an ADR might not be necessary (though documenting the rationale in code comments or commit messages is still good practice).

## ADR Format

All ADRs should follow the template defined in [ADR_TEMPLATE.md](./ADR_TEMPLATE.md).

## ADR Lifecycle

1.  **Proposed:** An ADR is drafted and submitted for discussion/review (e.g., via a Pull Request).
2.  **Accepted:** After discussion and agreement by the relevant stakeholders (e.g., tech lead, team), the ADR is merged and its status is marked as "Accepted."
3.  **Rejected/Withdrawn (Optional):** If an ADR is not accepted, it can be marked as such or withdrawn.
4.  **Deprecated/Superseded:** If a new decision makes an existing ADR obsolete, the old ADR's status is updated to "Deprecated" or "Superseded by ADR-XXX" (linking to the new ADR). The new ADR should also reference the one it supersedes.

## Naming Convention

ADRs should be named sequentially, e.g., `001-decision-summary.md`, `002-another-decision.md`. The title within the ADR document should be more descriptive.

## Current ADRs

*   *(This section should list links to actual ADRs as they are created)*
*   Example: `[000-use-of-adrs.md](./000-use-of-adrs.md)` (An ADR about using ADRs itself!)
