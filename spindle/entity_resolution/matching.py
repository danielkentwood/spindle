"""Semantic matching for entity resolution.

LLM-based duplicate matching within blocks using BAML client to call Claude
for intelligent duplicate detection.
"""

from typing import Any, Dict, List
import json
import time

import baml_py

from spindle.baml_client import b
from spindle.baml_client.types import (
    EntityForMatching,
    EdgeForMatching,
    EntityMatch as BamlEntityMatch,
    EdgeMatch as BamlEdgeMatch,
)

from spindle.entity_resolution.config import ResolutionConfig
from spindle.entity_resolution.models import EntityMatch, EdgeMatch
from spindle.entity_resolution.utils import (
    _record_resolution_event,
    _extract_model_from_collector,
    _extract_metrics_from_collector,
)


class SemanticMatcher:
    """LLM-based duplicate matching within blocks.
    
    Uses BAML client to call Claude for intelligent duplicate detection.
    """
    
    def __init__(self, config: ResolutionConfig):
        """Initialize semantic matcher with configuration.
        
        Args:
            config: ResolutionConfig with matching parameters
        """
        self.config = config
    
    def match_entities(
        self,
        block: List[Dict[str, Any]],
        context: str = ""
    ) -> List[EntityMatch]:
        """Find duplicate entities within a block using LLM.
        
        Args:
            block: List of entity dictionaries to match
            context: Optional context about the domain/ontology
        
        Returns:
            List of EntityMatch objects for duplicates found
        """
        _record_resolution_event(
            "matching.entities.start",
            {
                "block_size": len(block),
                "has_context": bool(context)
            }
        )
        
        if len(block) < 2:
            return []
        
        # Convert entities to BAML format
        entities_for_matching = []
        for entity in block:
            # Serialize attributes
            attrs = entity.get('custom_atts', {})
            if isinstance(attrs, str):
                attrs_str = attrs
            else:
                attrs_str = json.dumps(attrs)
            
            entities_for_matching.append(EntityForMatching(
                id=entity['name'],
                type=entity.get('type', ''),
                description=entity.get('description', ''),
                attributes=attrs_str
            ))
        
        # Process in batches if block is large
        all_matches = []
        batch_size = self.config.batch_size
        
        total_tokens = 0
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0
        models_used = []  # Track models used across batches
        
        for i in range(0, len(entities_for_matching), batch_size):
            batch = entities_for_matching[i:i + batch_size]
            
            try:
                # Track LLM metrics using BAML collector
                collector = baml_py.baml_py.Collector(f"entity-matching-batch-{i}")
                start_time = time.perf_counter()
                
                # Call BAML function with collector
                result = b.with_options(collector=collector).MatchEntities(
                    entities=batch,
                    context=context
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                total_latency_ms += latency_ms
                
                # Extract metrics from collector
                metrics = _extract_metrics_from_collector(collector)
                batch_tokens = metrics["total_tokens"]
                batch_cost = metrics["total_cost"]
                batch_input_tokens = metrics.get("input_tokens") or 0
                batch_output_tokens = metrics.get("output_tokens") or 0
                total_tokens += batch_tokens
                total_cost += batch_cost
                if batch_input_tokens:
                    total_input_tokens += batch_input_tokens
                if batch_output_tokens:
                    total_output_tokens += batch_output_tokens
                
                # Extract model information from collector
                # Falls back to "CustomSonnet4" (the client configured for MatchEntities in BAML)
                model = _extract_model_from_collector(collector) or "CustomSonnet4"
                if model:
                    models_used.append(model)

                # Fallback cost estimation if provider did not supply a cost
                if batch_cost == 0.0:
                    try:
                        from spindle.llm_pricing import estimate_cost_usd
                        est = estimate_cost_usd(model, batch_input_tokens, batch_output_tokens)
                        if est is not None:
                            total_cost += est
                    except Exception:
                        pass
                
                # Convert BAML matches to our EntityMatch format
                # and filter by confidence threshold
                for match in result.matches:
                    # Convert confidence level to numeric score
                    confidence_score = self._confidence_level_to_score(match.confidence_level)
                    
                    if confidence_score >= self.config.matching_threshold:
                        all_matches.append(EntityMatch(
                            entity1_id=match.entity1_id,
                            entity2_id=match.entity2_id,
                            is_duplicate=match.is_duplicate,
                            confidence=confidence_score,
                            reasoning=match.reasoning
                        ))
            except Exception as e:
                _record_resolution_event(
                    "matching.entities.error",
                    {
                        "batch_index": i,
                        "batch_size": len(batch),
                        "error": str(e)
                    }
                )
                # Continue with next batch
                continue
        
        # Determine primary model (most common, or first if all different)
        primary_model = None
        if models_used:
            from collections import Counter
            model_counts = Counter(models_used)
            primary_model = model_counts.most_common(1)[0][0]
        
        _record_resolution_event(
            "matching.entities.complete",
            {
                "block_size": len(block),
                "matches_found": len(all_matches),
                "input_tokens": total_input_tokens if total_input_tokens > 0 else None,
                "output_tokens": total_output_tokens if total_output_tokens > 0 else None,
                "cost": total_cost,
                "latency_ms": total_latency_ms,
                "model": primary_model,
            }
        )
        
        return all_matches
    
    def match_edges(
        self,
        block: List[Dict[str, Any]],
        context: str = ""
    ) -> List[EdgeMatch]:
        """Find duplicate edges within a block using LLM.
        
        Args:
            block: List of edge dictionaries to match
            context: Optional context about the domain/ontology
        
        Returns:
            List of EdgeMatch objects for duplicates found
        """
        _record_resolution_event(
            "matching.edges.start",
            {
                "block_size": len(block),
                "has_context": bool(context)
            }
        )
        
        if len(block) < 2:
            return []
        
        # Convert edges to BAML format
        edges_for_matching = []
        for edge in block:
            # Create edge ID
            edge_id = f"{edge['subject']}|{edge['predicate']}|{edge['object']}"
            
            # Summarize evidence
            evidence = edge.get('supporting_evidence', [])
            if isinstance(evidence, str):
                try:
                    evidence = json.loads(evidence)
                except json.JSONDecodeError:
                    evidence = []
            
            evidence_summary = ""
            if isinstance(evidence, list) and evidence:
                source_count = len(evidence)
                evidence_summary = f"{source_count} source(s)"
                
                # Add sample text from first source
                first_source = evidence[0]
                if isinstance(first_source, dict) and first_source.get('spans'):
                    spans = first_source['spans']
                    if spans and isinstance(spans, list):
                        first_span = spans[0]
                        if isinstance(first_span, dict) and first_span.get('text'):
                            sample = first_span['text'][:100]
                            evidence_summary += f": {sample}..."
            
            edges_for_matching.append(EdgeForMatching(
                id=edge_id,
                subject=edge['subject'],
                predicate=edge['predicate'],
                object=edge['object'],
                evidence_summary=evidence_summary
            ))
        
        # Process in batches if block is large
        all_matches = []
        batch_size = self.config.batch_size
        
        total_tokens = 0
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0
        models_used = []  # Track models used across batches
        
        for i in range(0, len(edges_for_matching), batch_size):
            batch = edges_for_matching[i:i + batch_size]
            
            try:
                # Track LLM metrics using BAML collector
                collector = baml_py.baml_py.Collector(f"edge-matching-batch-{i}")
                start_time = time.perf_counter()
                
                # Call BAML function with collector
                result = b.with_options(collector=collector).MatchEdges(
                    edges=batch,
                    context=context
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                total_latency_ms += latency_ms
                
                # Extract metrics from collector
                metrics = _extract_metrics_from_collector(collector)
                batch_tokens = metrics["total_tokens"]
                batch_cost = metrics["total_cost"]
                batch_input_tokens = metrics.get("input_tokens") or 0
                batch_output_tokens = metrics.get("output_tokens") or 0
                total_tokens += batch_tokens
                total_cost += batch_cost
                if batch_input_tokens:
                    total_input_tokens += batch_input_tokens
                if batch_output_tokens:
                    total_output_tokens += batch_output_tokens
                
                # Extract model information from collector
                # Falls back to "CustomSonnet4" (the client configured for MatchEdges in BAML)
                model = _extract_model_from_collector(collector) or "CustomSonnet4"
                if model:
                    models_used.append(model)

                # Fallback cost estimation if provider did not supply a cost
                if batch_cost == 0.0:
                    try:
                        from spindle.llm_pricing import estimate_cost_usd
                        est = estimate_cost_usd(model, batch_input_tokens, batch_output_tokens)
                        if est is not None:
                            total_cost += est
                    except Exception:
                        pass
                
                # Convert BAML matches to our EdgeMatch format
                # and filter by confidence threshold
                for match in result.matches:
                    # Convert confidence level to numeric score
                    confidence_score = self._confidence_level_to_score(match.confidence_level)
                    
                    if confidence_score >= self.config.matching_threshold:
                        all_matches.append(EdgeMatch(
                            edge1_id=match.edge1_id,
                            edge2_id=match.edge2_id,
                            is_duplicate=match.is_duplicate,
                            confidence=confidence_score,
                            reasoning=match.reasoning
                        ))
            except Exception as e:
                _record_resolution_event(
                    "matching.edges.error",
                    {
                        "batch_index": i,
                        "batch_size": len(batch),
                        "error": str(e)
                    }
                )
                # Continue with next batch
                continue
        
        # Determine primary model (most common, or first if all different)
        primary_model = None
        if models_used:
            from collections import Counter
            model_counts = Counter(models_used)
            primary_model = model_counts.most_common(1)[0][0]
        
        _record_resolution_event(
            "matching.edges.complete",
            {
                "block_size": len(block),
                "matches_found": len(all_matches),
                "input_tokens": total_input_tokens if total_input_tokens > 0 else None,
                "output_tokens": total_output_tokens if total_output_tokens > 0 else None,
                "cost": total_cost,
                "latency_ms": total_latency_ms,
                "model": primary_model,
            }
        )
        
        return all_matches
    
    def _confidence_level_to_score(self, level: str) -> float:
        """Convert confidence level string to numeric score.
        
        Args:
            level: Confidence level ('high', 'medium', or 'low')
        
        Returns:
            Numeric confidence score (0.0-1.0)
        """
        level_map = {
            'high': 0.95,
            'medium': 0.75,
            'low': 0.50
        }
        return level_map.get(level.lower(), 0.50)

