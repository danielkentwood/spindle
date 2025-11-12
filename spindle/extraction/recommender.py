"""
Ontology Recommender Module

This module provides ontology recommendation functionality for Spindle.

Key Components:
- OntologyRecommender: Automatically recommend ontologies by analyzing text,
  and conservatively extend existing ontologies when processing new sources
"""

from typing import List, Dict, Any, Optional, Tuple
import time
import baml_py
from spindle.baml_client import b
from spindle.baml_client.types import (
    ExtractionResult,
    Ontology,
    OntologyExtension,
    OntologyRecommendation,
    Triple,
)
from spindle.extraction.helpers import (
    _record_ontology_event,
    _extract_model_from_collector,
)

try:
    from spindle.llm_config import (
        LLMConfig,
        detect_available_auth,
        create_baml_env_overrides,
    )

    LLM_CONFIG_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    LLMConfig = None
    detect_available_auth = None
    create_baml_env_overrides = None
    LLM_CONFIG_AVAILABLE = False


class OntologyRecommender:
    """
    Service for recommending ontologies based on text analysis.
    
    This class wraps the BAML ontology recommendation function and provides
    a simple interface for automatically generating ontologies suitable for
    knowledge graph extraction from text.
    
    Uses principle-based ontology design with configurable scope levels
    instead of hard numerical limits.

    Supports optional LLM configuration for authenticated access, with automatic
    credential detection when explicit configuration is not supplied.
    """

    def __init__(
        self,
        llm_config: Optional["LLMConfig"] = None,
        auto_detect_auth: bool = True,
    ):
        """
        Initialize the ontology recommender with optional LLM configuration.

        Args:
            llm_config: Optional LLMConfig providing authentication details. If None
                and auto_detect_auth is True, credentials will be auto-detected.
            auto_detect_auth: Whether to auto-detect authentication when llm_config
                is not provided.
        """
        self.llm_config = llm_config
        self.auto_detect_auth = auto_detect_auth
        self._baml_env_overrides: Optional[Dict[str, str]] = None
        self._configure_baml_client(self.llm_config, self.auto_detect_auth)

    def _configure_baml_client(
        self,
        llm_config: Optional["LLMConfig"],
        auto_detect_auth: bool,
    ) -> None:
        """Configure BAML client environment overrides based on LLM config."""
        if not LLM_CONFIG_AVAILABLE:
            self._baml_env_overrides = None
            return

        config = llm_config
        if config is None and auto_detect_auth and detect_available_auth is not None:
            config = detect_available_auth()

        if config is None or create_baml_env_overrides is None:
            self._baml_env_overrides = None
            return

        self.llm_config = config
        self._baml_env_overrides = create_baml_env_overrides(config)

    def _get_baml_client(self):
        """Return the configured BAML client."""
        if self._baml_env_overrides:
            return b.with_options(env=self._baml_env_overrides)
        return b
    
    def recommend(
        self,
        text: str,
        scope: str = "balanced"
    ) -> OntologyRecommendation:
        """
        Analyze text and recommend an appropriate ontology.
        
        This method examines the provided text to infer its overarching
        purpose/goal and domain, then recommends entity types and relation
        types that would be most suitable for extracting knowledge from
        this and similar texts.
        
        The ontology is designed using principled guidelines about granularity
        and abstraction level, rather than hard numerical limits. The LLM
        determines the appropriate number of types based on the text's
        complexity and domain.
        
        Args:
            text: The text to analyze for ontology recommendation
            scope: Desired ontology scope, one of:
                  - "minimal": Essential concepts only (3-8 entity types, 4-10 relations)
                    Use for quick extraction, simple queries, broad patterns
                  - "balanced": Standard analysis (6-12 entity types, 8-15 relations)
                    Use for general-purpose extraction (DEFAULT)
                  - "comprehensive": Detailed ontology (10-20 entity types, 12-25 relations)
                    Use for detailed analysis, domain expertise, research
        
        Returns:
            OntologyRecommendation containing:
                - ontology: The recommended Ontology object (ready for use
                           with SpindleExtractor)
                - text_purpose: Analysis of the text's overarching purpose
                - reasoning: Explanation of the recommendation choices
        
        Example:
            >>> recommender = OntologyRecommender()
            >>> text = "Alice works at TechCorp in San Francisco..."
            >>> 
            >>> # Get a balanced ontology (default)
            >>> recommendation = recommender.recommend(text)
            >>> 
            >>> # Or request a specific scope
            >>> minimal_rec = recommender.recommend(text, scope="minimal")
            >>> comprehensive_rec = recommender.recommend(text, scope="comprehensive")
            >>> 
            >>> print(recommendation.text_purpose)
            >>> extractor = SpindleExtractor(recommendation.ontology)
        """
        _record_ontology_event(
            "recommend.start",
            {
                "scope": scope,
                "text_length": len(text),
            },
        )
        try:
            # Track LLM metrics using BAML collector
            collector = baml_py.baml_py.Collector("ontology-recommendation-collector")
            start_time = time.perf_counter()
            
            client = self._get_baml_client()
            result = client.with_options(collector=collector).RecommendOntology(
                text=text,
                scope=scope
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract metrics from collector
            total_tokens = getattr(collector, 'total_tokens', 0)
            total_cost = getattr(collector, 'total_cost', 0.0)
            input_tokens = getattr(collector, 'input_tokens', None)
            output_tokens = getattr(collector, 'output_tokens', None)
            if input_tokens is None and output_tokens is None:
                if hasattr(collector, 'logs') and collector.logs:
                    last_log = collector.logs[-1] if collector.logs else None
                    if last_log:
                        input_tokens = getattr(last_log, 'input_tokens', None)
                        output_tokens = getattr(last_log, 'output_tokens', None)
            
            # Extract model information
            model = _extract_model_from_collector(collector)
        except Exception as exc:
            _record_ontology_event(
                "recommend.error",
                {
                    "scope": scope,
                    "error": str(exc),
                },
            )
            raise

        _record_ontology_event(
            "recommend.complete",
            {
                "scope": scope,
                "entity_type_count": len(result.ontology.entity_types),
                "relation_type_count": len(result.ontology.relation_types),
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": total_cost,
                "latency_ms": latency_ms,
                "model": model,
            },
        )
        return result
    
    def recommend_and_extract(
        self,
        text: str,
        source_name: str,
        source_url: Optional[str] = None,
        scope: str = "balanced",
        existing_triples: List[Triple] = None
    ) -> Tuple[OntologyRecommendation, ExtractionResult]:
        """
        Recommend an ontology and immediately use it to extract triples.
        
        This is a convenience method that combines ontology recommendation
        and triple extraction in one step. Useful when you don't have a
        pre-defined ontology and want to let the system automatically
        determine the best structure for your text.
        
        Args:
            text: The text to analyze and extract from
            source_name: Name or identifier of the source document
            source_url: Optional URL of the source document
            scope: Desired ontology scope ("minimal", "balanced", or "comprehensive")
            existing_triples: Optional list of previously extracted triples
                             for entity consistency
        
        Returns:
            Tuple of (OntologyRecommendation, ExtractionResult):
                - OntologyRecommendation with the generated ontology
                - ExtractionResult with extracted triples using that ontology
        
        Example:
            >>> recommender = OntologyRecommender()
            >>> text = "Alice works at TechCorp..."
            >>> 
            >>> # Use default balanced scope
            >>> recommendation, extraction = recommender.recommend_and_extract(
            ...     text=text,
            ...     source_name="example.txt"
            ... )
            >>> 
            >>> # Or request comprehensive scope
            >>> recommendation, extraction = recommender.recommend_and_extract(
            ...     text=complex_paper,
            ...     source_name="research.pdf",
            ...     scope="comprehensive"
            ... )
            >>> 
            >>> print(f"Purpose: {recommendation.text_purpose}")
            >>> print(f"Extracted {len(extraction.triples)} triples")
        """
        _record_ontology_event(
            "recommend_and_extract.start",
            {
                "scope": scope,
                "text_length": len(text),
            },
        )
        try:
            # First, recommend the ontology
            recommendation = self.recommend(
                text=text,
                scope=scope
            )

            # Then, use the recommended ontology to extract triples
            # Import here to avoid circular import
            from spindle.extraction.extractor import SpindleExtractor
            
            extractor = SpindleExtractor(
                recommendation.ontology,
                llm_config=self.llm_config,
                auto_detect_auth=self.auto_detect_auth,
            )
            extraction_result = extractor.extract(
                text=text,
                source_name=source_name,
                source_url=source_url,
                existing_triples=existing_triples
            )
        except Exception as exc:
            _record_ontology_event(
                "recommend_and_extract.error",
                {
                    "scope": scope,
                    "source_name": source_name,
                    "error": str(exc),
                },
            )
            raise

        _record_ontology_event(
            "recommend_and_extract.complete",
            {
                "scope": scope,
                "source_name": source_name,
                "triple_count": len(extraction_result.triples),
            },
        )
        return recommendation, extraction_result
    
    def analyze_extension(
        self,
        text: str,
        current_ontology: Ontology,
        scope: str = "balanced"
    ) -> OntologyExtension:
        """
        Analyze whether an existing ontology needs to be extended for new text.
        
        This method conservatively analyzes if the current ontology is sufficient
        to extract critical information from new text, or if it needs extension.
        
        CONSERVATIVE APPROACH: The analysis defaults to NO extension. Extensions
        are only recommended if failing to extend would result in losing critical
        information that cannot be captured with existing types.
        
        Args:
            text: The new text to analyze
            current_ontology: The existing Ontology to potentially extend
            scope: The scope level to consider ("minimal", "balanced", "comprehensive")
                  This affects how conservative the extension analysis is.
        
        Returns:
            OntologyExtension containing:
                - needs_extension: Boolean indicating if extension is needed
                - new_entity_types: List of new entity types to add (empty if none)
                - new_relation_types: List of new relation types to add (empty if none)
                - critical_information_at_risk: Description of what would be lost
                - reasoning: Detailed explanation of the decision
        
        Example:
            >>> recommender = OntologyRecommender()
            >>> # Existing ontology for business domain
            >>> ontology = create_ontology(business_entities, business_relations)
            >>> 
            >>> # Analyze new text from medical domain
            >>> medical_text = "Dr. Chen prescribed Medication A for the patient..."
            >>> extension = recommender.analyze_extension(
            ...     text=medical_text,
            ...     current_ontology=ontology,
            ...     scope="balanced"
            ... )
            >>> 
            >>> if extension.needs_extension:
            ...     print(f"Extension needed: {extension.critical_information_at_risk}")
            ...     print(f"New types: {[et.name for et in extension.new_entity_types]}")
            ... else:
            ...     print(f"No extension needed: {extension.reasoning}")
        """
        client = self._get_baml_client()
        result = client.AnalyzeOntologyExtension(
            text=text,
            current_ontology=current_ontology,
            scope=scope
        )
        
        return result
    
    def extend_ontology(
        self,
        current_ontology: Ontology,
        extension: OntologyExtension
    ) -> Ontology:
        """
        Apply an ontology extension to create an extended ontology.
        
        This creates a new Ontology object by combining the current ontology
        with the new types from the extension analysis.
        
        Args:
            current_ontology: The existing Ontology
            extension: The OntologyExtension from analyze_extension()
        
        Returns:
            A new Ontology object with the extensions applied
        
        Example:
            >>> extension = recommender.analyze_extension(text, ontology)
            >>> if extension.needs_extension:
            ...     extended_ontology = recommender.extend_ontology(ontology, extension)
            ...     print(f"Original: {len(ontology.entity_types)} types")
            ...     print(f"Extended: {len(extended_ontology.entity_types)} types")
        """
        # Combine existing and new entity types
        all_entity_types = list(current_ontology.entity_types) + list(extension.new_entity_types)
        
        # Combine existing and new relation types
        all_relation_types = list(current_ontology.relation_types) + list(extension.new_relation_types)
        
        # Create and return the extended ontology
        return Ontology(
            entity_types=all_entity_types,
            relation_types=all_relation_types
        )
    
    def analyze_and_extend(
        self,
        text: str,
        current_ontology: Ontology,
        scope: str = "balanced",
        auto_apply: bool = True
    ) -> Tuple[OntologyExtension, Optional[Ontology]]:
        """
        Analyze extension needs and optionally apply them in one step.
        
        This is a convenience method that combines analyze_extension() and
        extend_ontology() with optional automatic application.
        
        Args:
            text: The new text to analyze
            current_ontology: The existing Ontology
            scope: The scope level for analysis
            auto_apply: If True, automatically apply extensions and return new ontology.
                       If False, return None as second element (user must apply manually).
        
        Returns:
            Tuple of (OntologyExtension, Optional[Ontology]):
                - OntologyExtension with the analysis results
                - Extended Ontology if auto_apply=True and needs_extension=True, else None
        
        Example:
            >>> # Automatic application
            >>> extension, new_ontology = recommender.analyze_and_extend(
            ...     text=new_text,
            ...     current_ontology=ontology,
            ...     auto_apply=True
            ... )
            >>> 
            >>> if new_ontology:
            ...     # Extension was needed and applied
            ...     extractor = SpindleExtractor(new_ontology)
            ... else:
            ...     # No extension needed, use original
            ...     extractor = SpindleExtractor(ontology)
        """
        extension = self.analyze_extension(text, current_ontology, scope)
        
        if extension.needs_extension and auto_apply:
            extended_ontology = self.extend_ontology(current_ontology, extension)
            return extension, extended_ontology
        
        return extension, None

