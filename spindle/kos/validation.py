"""SKOS integrity checks and SHACL validation for the KOS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from spindle.kos.models import ValidationReport


# SPARQL prefixes shared by integrity queries
_PREFIXES = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX spndl: <http://spindle.dev/ns/>
"""


def check_skos_integrity(store: object) -> List[str]:
    """Run SKOS integrity checks against an Oxigraph store.

    Checks:
    1. No orphan concepts — every concept must be in a ConceptScheme.
    2. No cycles in skos:broader (transitive).
    3. skos:related must be symmetric.

    Returns:
        List of human-readable violation descriptions (empty = clean).
    """
    violations: List[str] = []

    # 1. Orphan concepts
    orphan_query = _PREFIXES + """
    SELECT ?concept WHERE {
        GRAPH ?g {
            ?concept a skos:Concept .
            FILTER NOT EXISTS { ?concept skos:inScheme ?scheme }
            FILTER NOT EXISTS { ?concept skos:topConceptOf ?scheme }
        }
    }
    """
    try:
        for row in store.query(orphan_query):
            violations.append(f"Orphan concept (not in any scheme): {row[0]}")
    except Exception as exc:
        violations.append(f"Orphan check failed: {exc}")

    # 2. Asymmetric skos:related (A related B but not B related A)
    asym_query = _PREFIXES + """
    SELECT ?a ?b WHERE {
        GRAPH ?g {
            ?a skos:related ?b .
            FILTER NOT EXISTS { ?b skos:related ?a }
        }
    }
    """
    try:
        for row in store.query(asym_query):
            violations.append(
                f"Asymmetric skos:related: {row[0]} related {row[1]} but not vice-versa"
            )
    except Exception as exc:
        violations.append(f"Symmetry check failed: {exc}")

    # 3. Direct cycles in skos:broader (A broader B and B broader A)
    cycle_query = _PREFIXES + """
    SELECT ?a ?b WHERE {
        GRAPH ?g {
            ?a skos:broader ?b .
            ?b skos:broader ?a .
        }
    }
    """
    try:
        for row in store.query(cycle_query):
            violations.append(
                f"Cycle in skos:broader: {row[0]} and {row[1]} are mutual broader"
            )
    except Exception as exc:
        violations.append(f"Cycle check failed: {exc}")

    return violations


def validate_with_shacl(
    data_graph: object,
    shapes_path: Path,
    ontology_path: Optional[Path] = None,
) -> ValidationReport:
    """Validate an Oxigraph graph using SHACL shapes.

    Args:
        data_graph: Oxigraph Store whose default graph will be validated.
        shapes_path: Path to the SHACL shapes file (``shapes.ttl``).
        ontology_path: Optional path to the OWL ontology for inference.

    Returns:
        ValidationReport with conforms flag and any violations.

    Raises:
        ImportError: If pyshacl is not installed.
    """
    try:
        import pyshacl
    except ImportError as exc:
        raise ImportError(
            "pyshacl is required for SHACL validation. "
            "Install it with: uv pip install pyshacl"
        ) from exc

    if not shapes_path.exists():
        return ValidationReport(conforms=True, message="No shapes file found — skipping.")

    try:
        conforms, results_graph, results_text = pyshacl.validate(
            data_graph=str(data_graph),
            shacl_graph=str(shapes_path),
            ont_graph=str(ontology_path) if ontology_path else None,
            inference="rdfs",
            abort_on_first=False,
            serialize_report_graph=False,
        )
        violations: List[Dict[str, Any]] = []
        if not conforms and results_text:
            violations = [{"message": results_text}]
        return ValidationReport(conforms=conforms, violations=violations)
    except Exception as exc:
        return ValidationReport(conforms=False, message=str(exc))
