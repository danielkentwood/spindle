"""
Spindle Extractor Module

This module provides the core triple extraction functionality for Spindle.

Key Components:
- SpindleExtractor: Extract triples from text using a predefined ontology
"""

from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
from datetime import datetime
import time
import baml_py
from spindle.baml_client import b
from spindle.baml_client.async_client import b as async_b
from spindle.baml_client.types import (
    ExtractionResult,
    Ontology,
    SourceMetadata,
    Triple,
)
from spindle.extraction.helpers import (
    _record_extractor_event,
    _extract_model_from_collector,
    _compute_all_span_indices,
)
from spindle.extraction.recommender import OntologyRecommender

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


class SpindleExtractor:
    """
    Main interface for extracting knowledge graph triples from text.
    
    This class wraps the BAML extraction function and provides a simple
    interface for incremental triple extraction with entity consistency.
    Each extracted triple includes source metadata, supporting text spans,
    and an extraction datetime (set automatically in post-processing).
    
    Supports multiple authentication methods for LLM access via optional LLM
    configuration or automatic detection of available credentials.

    If no ontology is provided at initialization, the extractor will
    automatically recommend one based on the text when extract() is called.
    """
    
    def __init__(
        self,
        ontology: Optional[Ontology] = None,
        ontology_scope: str = "balanced",
        llm_config: Optional["LLMConfig"] = None,
        auto_detect_auth: bool = True,
    ):
        """
        Initialize the extractor with an ontology.
        
        Args:
            ontology: Optional Ontology object defining valid entity and relation types.
                     If None, an ontology will be automatically recommended from the
                     text when extract() is first called.
            ontology_scope: Scope for auto-recommended ontologies. One of:
                          - "minimal": Essential concepts only (3-8 entity types, 4-10 relations)
                          - "balanced": Standard analysis (6-12 entity types, 8-15 relations)
                          - "comprehensive": Detailed ontology (10-20 entity types, 12-25 relations)
                          Only used if ontology is None.
            llm_config: Optional LLMConfig providing explicit authentication details
                for LLM access. If None and auto_detect_auth is True, credentials
                will be auto-detected from the environment.
            auto_detect_auth: Whether to auto-detect authentication credentials when
                llm_config is not provided. Defaults to True.
        """
        self.ontology = ontology
        self.ontology_scope = ontology_scope
        self.llm_config = llm_config
        self.auto_detect_auth = auto_detect_auth
        self._baml_env_overrides: Optional[Dict[str, str]] = None

        self._configure_baml_client(self.llm_config, self.auto_detect_auth)
        self._ontology_recommender = (
            None
            if ontology is not None
            else OntologyRecommender(
                llm_config=self.llm_config,
                auto_detect_auth=self.auto_detect_auth,
            )
        )
    
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
        """Return the configured BAML sync client."""
        if self._baml_env_overrides:
            return b.with_options(env=self._baml_env_overrides)
        return b

    def _get_async_baml_client(self):
        """Return the configured BAML async client."""
        if self._baml_env_overrides:
            return async_b.with_options(env=self._baml_env_overrides)
        return async_b
    
    def extract(
        self,
        text: str,
        source_name: str,
        source_url: Optional[str] = None,
        existing_triples: List[Triple] = None,
        ontology_scope: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract triples from text using the configured or auto-recommended ontology.
        
        If no ontology was provided at initialization, one will be automatically
        recommended based on the text before extraction.
        
        Args:
            text: The text to extract triples from
            source_name: Name or identifier of the source document
            source_url: Optional URL of the source document
            existing_triples: Optional list of previously extracted triples
                            to maintain entity consistency. Duplicate triples
                            from different sources are allowed.
            ontology_scope: Override the default ontology scope for this extraction.
                          One of "minimal", "balanced", or "comprehensive".
                          Only used if no ontology was provided at init.
        
        Returns:
            ExtractionResult containing the extracted triples (with source
            metadata, supporting spans, and extraction datetime) and reasoning
        """
        if existing_triples is None:
            existing_triples = []

        _record_extractor_event(
            "extract.start",
            {
                "source_name": source_name,
                "source_url": source_url,
                "existing_triples": len(existing_triples),
            },
        )

        try:
            # Auto-recommend ontology if not provided
            if self.ontology is None:
                scope = ontology_scope or self.ontology_scope
                recommendation = self._ontology_recommender.recommend(
                    text=text,
                    scope=scope
                )
                self.ontology = recommendation.ontology

            # Create source metadata
            source_metadata = SourceMetadata(
                source_name=source_name,
                source_url=source_url
            )

            # Track LLM metrics using BAML collector
            collector = baml_py.baml_py.Collector("extraction-collector")
            start_time = time.perf_counter()
            
            # Call the BAML extraction function with collector
            baml_client = self._get_baml_client()
            result = baml_client.with_options(collector=collector).ExtractTriples(
                text=text,
                ontology=self.ontology,
                source_metadata=source_metadata,
                existing_triples=existing_triples
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract metrics from collector
            total_tokens = getattr(collector, 'total_tokens', 0)
            total_cost = getattr(collector, 'total_cost', 0.0)
            # Try to get input/output tokens if available
            input_tokens = getattr(collector, 'input_tokens', None)
            output_tokens = getattr(collector, 'output_tokens', None)
            if input_tokens is None and output_tokens is None:
                # Fallback: try to get from logs if available
                if hasattr(collector, 'logs') and collector.logs:
                    # Try to extract from last log entry
                    last_log = collector.logs[-1] if collector.logs else None
                    if last_log:
                        input_tokens = getattr(last_log, 'input_tokens', None)
                        output_tokens = getattr(last_log, 'output_tokens', None)
            
            # Extract model information
            model = _extract_model_from_collector(collector)

            # Post-processing: Set extraction datetime for all triples
            extraction_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            for triple in result.triples:
                triple.extraction_datetime = extraction_time

                # Post-processing: Compute character indices for supporting spans using batch processing
                triple.supporting_spans = _compute_all_span_indices(text, triple.supporting_spans)
            
            # Determine ontology scope for strategy grouping
            used_scope = ontology_scope or self.ontology_scope or "balanced"
        except Exception as exc:
            _record_extractor_event(
                "extract.error",
                {
                    "source_name": source_name,
                    "error": str(exc),
                },
            )
            raise

        _record_extractor_event(
            "extract.complete",
            {
                "source_name": source_name,
                "triple_count": len(result.triples),
                "ontology_scope": used_scope,
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": total_cost,
                "latency_ms": latency_ms,
                "model": model,
            },
        )
        return result
    
    async def extract_async(
        self,
        text: str,
        source_name: str,
        source_url: Optional[str] = None,
        existing_triples: List[Triple] = None,
        ontology_scope: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract triples from text using the configured or auto-recommended ontology (async version).
        
        This is the async version of extract(). It uses the async BAML client for
        non-blocking extraction, which is useful for batch processing and streaming.
        
        If no ontology was provided at initialization, one will be automatically
        recommended based on the text before extraction.
        
        Args:
            text: The text to extract triples from
            source_name: Name or identifier of the source document
            source_url: Optional URL of the source document
            existing_triples: Optional list of previously extracted triples
                            to maintain entity consistency. Duplicate triples
                            from different sources are allowed.
            ontology_scope: Override the default ontology scope for this extraction.
                          One of "minimal", "balanced", or "comprehensive".
                          Only used if no ontology was provided at init.
        
        Returns:
            ExtractionResult containing the extracted triples (with source
            metadata, supporting spans, and extraction datetime) and reasoning
        """
        if existing_triples is None:
            existing_triples = []

        _record_extractor_event(
            "extract_async.start",
            {
                "source_name": source_name,
                "source_url": source_url,
                "existing_triples": len(existing_triples),
            },
        )

        try:
            # Auto-recommend ontology if not provided
            if self.ontology is None:
                scope = ontology_scope or self.ontology_scope
                recommendation = self._ontology_recommender.recommend(
                    text=text,
                    scope=scope
                )
                self.ontology = recommendation.ontology

            # Create source metadata
            source_metadata = SourceMetadata(
                source_name=source_name,
                source_url=source_url
            )

            # Track LLM metrics using BAML collector
            collector = baml_py.baml_py.Collector("extraction-async-collector")
            start_time = time.perf_counter()
            
            # Call the async BAML extraction function with collector
            async_baml_client = self._get_async_baml_client()
            result = await async_baml_client.with_options(collector=collector).ExtractTriples(
                text=text,
                ontology=self.ontology,
                source_metadata=source_metadata,
                existing_triples=existing_triples
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

            # Post-processing: Set extraction datetime for all triples
            extraction_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            for triple in result.triples:
                triple.extraction_datetime = extraction_time

                # Post-processing: Compute character indices for supporting spans using batch processing
                triple.supporting_spans = _compute_all_span_indices(text, triple.supporting_spans)
            
            # Determine ontology scope for strategy grouping
            used_scope = ontology_scope or self.ontology_scope or "balanced"
        except Exception as exc:
            _record_extractor_event(
                "extract_async.error",
                {
                    "source_name": source_name,
                    "error": str(exc),
                },
            )
            raise

        _record_extractor_event(
            "extract_async.complete",
            {
                "source_name": source_name,
                "triple_count": len(result.triples),
                "ontology_scope": used_scope,
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": total_cost,
                "latency_ms": latency_ms,
                "model": model,
            },
        )
        return result
    
    async def extract_batch(
        self,
        texts: List[Tuple[str, str, Optional[str]]],
        existing_triples: List[Triple] = None,
        max_concurrent: int = 20,
        ontology_scope: Optional[str] = None
    ) -> List[ExtractionResult]:
        """
        Extract triples from multiple texts sequentially with entity consistency.
        
        This method processes texts one at a time (asynchronously) to maintain
        entity consistency. Triples extracted from earlier texts are automatically
        included in the existing_triples for later texts, ensuring consistent
        entity naming across the batch.
        
        Args:
            texts: List of tuples (text, source_name, source_url) to extract from.
                  source_url can be None.
            existing_triples: Optional initial list of previously extracted triples
                            to maintain entity consistency across the batch.
            max_concurrent: Maximum concurrent extractions (default: 20).
                          Currently used for internal async operations.
            ontology_scope: Override the default ontology scope for extractions.
                          One of "minimal", "balanced", or "comprehensive".
                          Only used if no ontology was provided at init.
        
        Returns:
            List of ExtractionResult objects, one per input text, in the same order
        """
        if existing_triples is None:
            existing_triples = []

        _record_extractor_event(
            "extract_batch.start",
            {
                "item_count": len(texts),
                "existing_triples": len(existing_triples),
            },
        )

        results = []
        accumulated_triples = list(existing_triples)

        try:
            # Process texts sequentially but asynchronously to maintain consistency
            for text, source_name, source_url in texts:
                # Pass a copy of accumulated_triples to avoid reference issues
                result = await self.extract_async(
                    text=text,
                    source_name=source_name,
                    source_url=source_url,
                    existing_triples=list(accumulated_triples),
                    ontology_scope=ontology_scope
                )
                results.append(result)
                # Accumulate triples from this extraction for next texts
                accumulated_triples.extend(result.triples)
        except Exception as exc:
            _record_extractor_event(
                "extract_batch.error",
                {
                    "item_count": len(texts),
                    "error": str(exc),
                },
            )
            raise

        _record_extractor_event(
            "extract_batch.complete",
            {
                "item_count": len(results),
                "total_triples": sum(len(res.triples) for res in results),
            },
        )
        return results
    
    async def extract_batch_stream(
        self,
        texts: List[Tuple[str, str, Optional[str]]],
        existing_triples: List[Triple] = None,
        max_concurrent: int = 20,
        ontology_scope: Optional[str] = None
    ) -> AsyncIterator[ExtractionResult]:
        """
        Extract triples from multiple texts, streaming results as they complete.
        
        This is an async generator that yields ExtractionResult objects as soon
        as each extraction completes. Results are yielded in the same order as
        the input texts, maintaining sequential consistency where triples from
        earlier texts are available to later texts.
        
        Args:
            texts: List of tuples (text, source_name, source_url) to extract from.
                  source_url can be None.
            existing_triples: Optional initial list of previously extracted triples
                            to maintain entity consistency across the batch.
            max_concurrent: Maximum concurrent extractions (default: 20).
                          Currently used for internal async operations.
            ontology_scope: Override the default ontology scope for extractions.
                          One of "minimal", "balanced", or "comprehensive".
                          Only used if no ontology was provided at init.
        
        Yields:
            ExtractionResult objects as they complete, in input order
        
        Example:
            >>> extractor = SpindleExtractor(ontology)
            >>> texts = [
            ...     ("Alice works at TechCorp", "doc1", None),
            ...     ("Bob manages Alice", "doc2", None)
            ... ]
            >>> async for result in extractor.extract_batch_stream(texts):
            ...     print(f"Extracted {len(result.triples)} triples from {result.triples[0].source.source_name}")
        """
        if existing_triples is None:
            existing_triples = []
        
        accumulated_triples = list(existing_triples)
        _record_extractor_event(
            "extract_batch_stream.start",
            {
                "item_count": len(texts),
                "existing_triples": len(existing_triples),
            },
        )

        try:
            # Process texts sequentially but asynchronously to maintain consistency
            for text, source_name, source_url in texts:
                # Pass a copy of accumulated_triples to avoid reference issues
                result = await self.extract_async(
                    text=text,
                    source_name=source_name,
                    source_url=source_url,
                    existing_triples=list(accumulated_triples),
                    ontology_scope=ontology_scope
                )
                _record_extractor_event(
                    "extract_batch_stream.item",
                    {
                        "source_name": source_name,
                        "triple_count": len(result.triples),
                    },
                )
                # Yield result as soon as it completes
                yield result
                # Accumulate triples from this extraction for next texts
                accumulated_triples.extend(result.triples)
        except Exception as exc:
            _record_extractor_event(
                "extract_batch_stream.error",
                {
                    "item_count": len(texts),
                    "error": str(exc),
                },
            )
            raise

        _record_extractor_event(
            "extract_batch_stream.complete",
            {
                "item_count": len(texts),
            },
        )

