"""Ontology synthesis: SKOS → OWL and OWL → SHACL.

Takes the consolidated kos.ttls as input, synthesises an OWL ontology from
the SKOS hierarchy, and optionally generates SHACL shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spindle.kos.service import KOSService


_SPINDLE_NS = "http://spindle.dev/ns/"
_SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
_OWL_NS = "http://www.w3.org/2002/07/owl#"
_RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
_SH_NS = "http://www.w3.org/ns/shacl#"


def synthesize_ontology(
    kos_service: "KOSService",
    output_path: Path,
    max_axioms_per_class: int = 10,
) -> Dict[str, Any]:
    """Derive an OWL ontology from the SKOS thesaurus.

    Each skos:Concept becomes an owl:Class; skos:broader/narrower relationships
    become rdfs:subClassOf axioms.

    Args:
        kos_service: Live KOSService containing the SKOS store.
        output_path: Destination path for the generated .owl (Turtle) file.
        max_axioms_per_class: Cap on axioms per class (prevents ontology bloat).

    Returns:
        Summary dict with counts.
    """
    if kos_service._store is None:
        return {"status": "skipped", "reason": "KOS store not loaded", "classes": 0}

    # Query SKOS store for concepts, labels, definitions, and broader hierarchy
    concepts_query = f"""
    PREFIX skos: <{_SKOS_NS}>
    PREFIX dct: <http://purl.org/dc/terms/>
    SELECT ?uri ?id ?label ?def WHERE {{
        GRAPH ?g {{
            ?uri a skos:Concept ;
                 skos:prefLabel ?label .
            OPTIONAL {{ ?uri dct:identifier ?id }}
            OPTIONAL {{ ?uri skos:definition ?def }}
        }}
    }}
    """
    broader_query = f"""
    PREFIX skos: <{_SKOS_NS}>
    SELECT ?child ?parent WHERE {{
        GRAPH ?g {{
            ?child skos:broader ?parent .
        }}
    }}
    """

    try:
        concept_rows = list(kos_service._store.query(concepts_query))
        broader_rows = list(kos_service._store.query(broader_query))
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "classes": 0}

    lines: List[str] = [
        f"@prefix owl:   <{_OWL_NS}> .",
        f"@prefix rdfs:  <{_RDFS_NS}> .",
        f"@prefix skos:  <{_SKOS_NS}> .",
        f"@prefix spndl: <{_SPINDLE_NS}> .",
        "",
        f"<{_SPINDLE_NS}ontology> a owl:Ontology .",
        "",
    ]

    class_count = 0
    axiom_count = 0
    per_class: Dict[str, int] = {}

    for row in concept_rows:
        uri = str(row[0])
        label = str(row[2])
        definition = str(row[3]) if row[3] else None

        lines.append(f"<{uri}>")
        lines.append("    a owl:Class ;")
        lines.append(f'    rdfs:label "{_escape(label)}"@en ;')
        if definition:
            lines.append(f'    rdfs:comment "{_escape(definition)}"@en ;')
        lines.append("    .")
        lines.append("")
        class_count += 1

    for row in broader_rows:
        child = str(row[0])
        parent = str(row[1])
        if per_class.get(child, 0) >= max_axioms_per_class:
            continue
        lines.append(f"<{child}> rdfs:subClassOf <{parent}> .")
        per_class[child] = per_class.get(child, 0) + 1
        axiom_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": "ok",
        "classes": class_count,
        "subclass_axioms": axiom_count,
        "output_path": str(output_path),
    }


def generate_shacl(
    ontology_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate SHACL node shapes from an OWL ontology file.

    Reads owl:Class declarations from the ontology and produces a
    minimal sh:NodeShape for each class that at least requires the
    expected rdf:type.

    Args:
        ontology_path: Path to the OWL Turtle file.
        output_path: Destination for generated shapes.ttl.
                     Defaults to ontology_path.parent / "shapes.ttl".

    Returns:
        Summary dict with counts.
    """
    if not ontology_path.exists():
        return {"status": "skipped", "reason": "ontology file not found", "shapes": 0}

    if output_path is None:
        output_path = ontology_path.parent / "shapes.ttl"

    text = ontology_path.read_text(encoding="utf-8")

    # Extract owl:Class declarations via simple regex (not full RDF parsing)
    import re
    class_uris = re.findall(r"<([^>]+)>\s*\n?\s*a owl:Class", text)

    lines: List[str] = [
        f"@prefix sh:    <{_SH_NS}> .",
        "@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        f"@prefix owl:   <{_OWL_NS}> .",
        "",
    ]

    shape_count = 0
    for uri in class_uris:
        slug = uri.rstrip("/").split("/")[-1]
        shape_uri = f"{_SPINDLE_NS}shape/{slug}"
        lines.append(f"<{shape_uri}>")
        lines.append("    a sh:NodeShape ;")
        lines.append(f"    sh:targetClass <{uri}> ;")
        lines.append(f"    sh:closed false ;")
        lines.append("    .")
        lines.append("")
        shape_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": "ok",
        "shapes": shape_count,
        "output_path": str(output_path),
    }


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
