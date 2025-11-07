// Process DAG Extraction Schemas and Function

enum ProcessStepType {
  ACTIVITY @description("Linear action that advances the process")
  DECISION @description("Branching evaluation that selects one of multiple paths")
  EVENT @description("External trigger or milestone that influences the process")
  PARALLEL_GATEWAY @description("Point where multiple activities run concurrently")
  SUBPROCESS @description("Reference to a nested process or reusable routine")
}

class EvidenceSpan {
  text string @description("Exact excerpt from the source that supports the step or dependency")
  start int? @description("Zero-based start index of the span within the text")
  end int? @description("End index (exclusive) of the span within the text")
}

class ProcessStep {
  step_id string @description("Stable identifier used for references and dependencies")
  title string @description("Short human-readable name for the step")
  summary string @description("Concise explanation of what happens during this step")
  step_type ProcessStepType @description("Category that characterises the behaviour of the step")
  actors string[] @description("People, roles, or systems responsible for executing the step")
  inputs string[] @description("Key inputs, prerequisites, or artifacts consumed by this step")
  outputs string[] @description("Artifacts, decisions, or state transitions produced by this step")
  duration string? @description("Estimated timing or duration information if mentioned")
  prerequisites string[] @description("Step identifiers that must be completed before this step starts")
  evidence EvidenceSpan[] @description("Supporting text spans drawn from the source material")
}

class ProcessDependency {
  from_step string @description("Identifier of the antecedent step")
  to_step string @description("Identifier of the dependent step")
  relation string @description("Nature of the dependency, e.g., 'precedes', 'blocks', 'enables'")
  condition string? @description("Conditional logic or guard that must be satisfied for the dependency")
  evidence EvidenceSpan[] @description("Supporting spans demonstrating this dependency in the text")
}

class ProcessGraph {
  process_name string? @description("Human-readable name or title for the overall process")
  scope string? @description("Short description of the process boundaries or context")
  primary_goal string @description("Statement summarising the overall objective of the process")
  start_step_ids string[] @description("Identifiers of steps that have no incoming dependencies")
  end_step_ids string[] @description("Identifiers of steps that have no outgoing dependencies")
  steps ProcessStep[] @description("All steps that participate in the process")
  dependencies ProcessDependency[] @description("Directed edges connecting steps to form a DAG")
  notes string[] @description("Additional remarks, open questions, or caveats captured by the model")
}

class ProcessExtractionIssue {
  code string @description("Machine-friendly identifier describing the issue")
  message string @description("Human-readable explanation of the issue or warning")
  related_step_ids string[] @description("Step identifiers relevant to this issue")
}

class ProcessExtractionResult {
  status "process_found" | "no_process" | "incomplete" @description("Outcome of the extraction attempt")
  graph ProcessGraph? @description("Structured representation of the process when available")
  reasoning string @description("Narrative justification covering modelling choices and DAG structure")
  issues ProcessExtractionIssue[] @description("Non-fatal problems encountered during extraction")
}

function ExtractProcessGraph(
  text: string,
  process_hint: string?,
  existing_graph: ProcessGraph?
) -> ProcessExtractionResult {
  client CustomFast
  prompt #"
    You are a process modelling specialist. Extract a directed acyclic graph (DAG) that captures the end-to-end process described in the provided text. Represent the process with well-defined steps and explicit dependencies.

    {{ _.role("user") }}

    TEXT:
    {{ text }}

    {% if process_hint %}
    PROCESS HINT:
    {{ process_hint }}
    {% endif %}

    {% if existing_graph %}
    EXISTING GRAPH CONTEXT:
    - Process name: {{ existing_graph.process_name or "Unknown" }}
    - Steps already known: {{ existing_graph.steps | map(attribute='step_id') | list }}
    When possible, reuse step identifiers from the existing graph to maintain continuity.
    {% endif %}

    OUTPUT REQUIREMENTS:
    1. Return status "no_process" if the text lacks procedural content.
    2. Always ensure dependency targets reference defined step identifiers.
    3. Identify start and end steps explicitly using computed arrays.
    4. Capture conditional flows, loops described as repeated checks, and parallel work using step types and dependency relation labels.
    5. Provide issues for ambiguous, conflicting, or incomplete information rather than guessing.
    6. Summarise reasoning with numbered bullet points that justify the chosen structure and highlight any assumptions.

    {{ ctx.output_format }}
  "#
}

