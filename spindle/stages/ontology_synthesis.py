"""Stage wrapper for ontology synthesis (SKOS → OWL → SHACL)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spindle.kos.service import KOSService


class OntologySynthesisStage:
    """Wraps synthesize_ontology + generate_shacl for spindle-eval Stage protocol.

    Args:
        kos_service: Live KOSService.
        output_dir: Directory to write ``ontology.owl`` and ``shapes.ttl``.
        max_axioms_per_class: Cap on subClassOf axioms per OWL class.
        generate_shacl: Whether to also run SHACL generation.
    """

    name: str = "ontology_synthesis"

    def __init__(
        self,
        kos_service: "KOSService",
        output_dir: Optional[Path] = None,
        max_axioms_per_class: int = 10,
        generate_shacl: bool = True,
    ) -> None:
        self._kos = kos_service
        self._output_dir = output_dir or kos_service._kos_dir
        self._max_axioms = max_axioms_per_class
        self._generate_shacl = generate_shacl

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Synthesise OWL from the loaded KOS and optionally generate SHACL.

        Returns:
            Combined summary dict.
        """
        from spindle.kos.synthesis import generate_shacl, synthesize_ontology

        owl_path = self._output_dir / "ontology.owl"
        result = synthesize_ontology(
            self._kos,
            output_path=owl_path,
            max_axioms_per_class=self._max_axioms,
        )

        if self._generate_shacl and result.get("status") == "ok":
            shapes_path = self._output_dir / "shapes.ttl"
            shacl_result = generate_shacl(owl_path, output_path=shapes_path)
            result["shacl"] = shacl_result

        return result

    def input_schema(self) -> Dict[str, Any]:
        return {"type": "null"}

    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "classes": {"type": "integer"},
                "shacl": {"type": "object"},
            },
        }

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        return self.run(**kwargs)
