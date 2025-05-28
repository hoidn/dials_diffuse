# System-Wide Shared Type Definitions [Type:System:1.0]

> This document is the authoritative source for system-wide shared data structures and type definitions used across multiple components and defined in IDL specifications.
>
> Define types here that are referenced by more than one major component or interface to ensure consistency and a single source of truth.
>
> Use a clear, language-agnostic syntax (like the TypeScript-inspired examples below) or a syntax that maps easily to your project's primary implementation language (e.g., Pydantic model definitions if it's a Python project).
>
> Remember to version your types if they undergo significant changes (e.g., `[Type:System:MyType:1.1]`).

---

## General Operation Types

```typescript
/**
 * Represents a generic operation outcome, often used as a return type
 * for methods that perform actions with potential success/failure.
 * [Type:System:OperationOutcome:1.0]
 */
interface OperationOutcome {
    /** Indicates if the operation was successful. */
    success: boolean;

    /** Optional message providing details about the outcome (e.g., error message, success confirmation). */
    message?: string;

    /** Optional data payload associated with the outcome (e.g., created entity, query results). */
    data?: Any; // 'Any' can be replaced with a more specific type or union if common patterns emerge

    /** Optional error code string for categorizing failures. */
    error_code?: string;

    /** Optional dictionary for additional metadata or diagnostic information. */
    metadata?: map<string, Any>;
}
```

## Configuration & Parameter Types

```typescript
/**
 * Defines common configuration parameters for a [YourSystemComponentType] component.
 * [Type:System:ComponentConfigOptions:1.0]
 */
interface ComponentConfigOptions {
    /** The network endpoint URL for the service. */
    endpoint_url: string; // Must be a valid URL format

    /** Maximum number of retries for transient failures. */
    max_retries: int; // Must be a non-negative integer

    /** Timeout duration in seconds for operations. */
    timeout_seconds?: int; // Optional, must be a positive integer if provided

    /** Feature flags or specific settings for the component. */
    settings?: map<string, boolean | string | int>;
}
```

## [Your Project Domain Specific Category, e.g., Task Execution Types]

```typescript
/**
 * Represents the status of a [YourProjectTask].
 * [Type:System:TaskStatus:1.0]
 */
type TaskStatus = "PENDING" | "IN_PROGRESS" | "COMPLETED" | "FAILED" | "CANCELLED";

/**
 * Input parameters for initiating a [YourProjectTask].
 * [Type:System:TaskRequestInput:1.0]
 */
interface TaskRequestInput {
    task_id: string; // Unique identifier for the task instance
    task_type: string; // Categorizes the task
    payload: Any; // Task-specific input data
    priority?: int; // Optional priority level
}
```

---

*Add more shared type definitions here as your project evolves, grouped by logical categories.*
