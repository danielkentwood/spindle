"""
Demo script for the Spindle Analytics Dashboard.

This script demonstrates:
1. Creating sample analytics observations with mocked LLM metrics
2. Persisting them to an analytics store
3. Launching the Streamlit dashboard to view the data
"""

import random
import sys
import subprocess
import os
from datetime import datetime, timedelta
from pathlib import Path

from spindle.analytics import AnalyticsStore
from spindle.analytics.schema import (
    ChunkWindowSummary,
    ContextStrategy,
    ContextWindowAssessment,
    DocumentMetadata,
    DocumentObservation,
    ObservabilitySignals,
    QuantileSummary,
    RiskLevel,
    SemanticSegmentSummary,
    ServiceEventRecord,
    SourceType,
    StructuralMetrics,
)
from spindle.observability import get_event_recorder
from spindle.observability.storage import EventLogStore


def generate_sample_observation(
    document_id: str,
    source_uri: str,
    token_count: int,
    chunk_count: int,
    strategy: ContextStrategy,
    risk: RiskLevel,
    ingested_at: datetime,
) -> DocumentObservation:
    """Generate a sample DocumentObservation with realistic data."""
    
    # Generate chunk token distribution
    avg_tokens_per_chunk = token_count // chunk_count
    tokens_per_chunk = [
        max(1, avg_tokens_per_chunk + random.randint(-avg_tokens_per_chunk // 2, avg_tokens_per_chunk // 2))
        for _ in range(chunk_count)
    ]
    # Ensure total matches approximately
    total_adjustment = token_count - sum(tokens_per_chunk)
    if total_adjustment != 0:
        tokens_per_chunk[0] += total_adjustment
    
    # Build chunk token summary
    chunk_token_summary = QuantileSummary(
        minimum=float(min(tokens_per_chunk)),
        maximum=float(max(tokens_per_chunk)),
        median=float(sorted(tokens_per_chunk)[len(tokens_per_chunk) // 2]),
        mean=float(sum(tokens_per_chunk) / len(tokens_per_chunk)),
        p95=float(sorted(tokens_per_chunk)[int(len(tokens_per_chunk) * 0.95)]),
    )
    
    # Build structural metrics
    structural = StructuralMetrics(
        token_count=token_count,
        character_count=token_count * 5,  # Rough estimate
        page_count=chunk_count // 2 if chunk_count > 1 else 1,
        section_count=chunk_count // 3 if chunk_count > 2 else 1,
        average_tokens_per_section=token_count / max(1, chunk_count // 3),
        chunk_count=chunk_count,
        chunk_token_summary=chunk_token_summary,
    )
    
    # Build chunk windows for different window sizes
    chunk_windows = []
    for window_size in [2, 3, 5]:
        if chunk_count >= window_size:
            window_tokens = [
                sum(tokens_per_chunk[i:i + window_size])
                for i in range(len(tokens_per_chunk) - window_size + 1)
            ]
            if window_tokens:
                window_summary = QuantileSummary(
                    minimum=float(min(window_tokens)),
                    maximum=float(max(window_tokens)),
                    median=float(sorted(window_tokens)[len(window_tokens) // 2]),
                    mean=float(sum(window_tokens) / len(window_tokens)),
                    p95=float(sorted(window_tokens)[int(len(window_tokens) * 0.95)]),
                )
                
                # Determine risk based on max tokens vs budget
                max_window_tokens = int(window_summary.maximum)
                window_risk = RiskLevel.LOW
                if max_window_tokens > 12000:
                    window_risk = RiskLevel.HIGH
                elif max_window_tokens > 8000:
                    window_risk = RiskLevel.MEDIUM
                
                chunk_windows.append(
                    ChunkWindowSummary(
                        window_size=window_size,
                        token_summary=window_summary,
                        overlap_tokens=None,
                        overlap_ratio=None,
                        cross_chunk_link_rate=random.uniform(0.1, 0.4),
                        context_limit_risk=window_risk,
                    )
                )
    
    # Build context assessment
    context = ContextWindowAssessment(
        recommended_strategy=strategy,
        supporting_risk=risk,
        estimated_token_usage=token_count if strategy == ContextStrategy.DOCUMENT else max(
            (w.token_summary.maximum for w in chunk_windows), default=token_count
        ),
        target_token_budget=12000,
    )
    
    # Build semantic segments
    segments = SemanticSegmentSummary(
        segment_boundaries=[0, token_count // 3, 2 * token_count // 3],
        segment_token_summary=chunk_token_summary,
        embedding_dispersion=random.uniform(0.3, 0.8),
        topic_transition_score=random.uniform(0.2, 0.7),
    )
    
    # Build observability signals
    service_events = [
        ServiceEventRecord(
            timestamp=ingested_at + timedelta(seconds=i),
            service="ingestion.pipeline",
            name=f"stage.{stage}",
            payload={"duration_ms": random.uniform(100, 1000)},
        )
        for i, stage in enumerate(["load", "chunk", "extract", "store"])
    ]
    
    observability = ObservabilitySignals(
        service_events=service_events,
        error_signals=[],
        latency_breakdown={
            "load": random.uniform(200, 500),
            "chunk": random.uniform(300, 800),
            "extract": random.uniform(1000, 3000),
            "store": random.uniform(100, 400),
        },
    )
    
    # Build metadata
    metadata = DocumentMetadata(
        document_id=document_id,
        source_uri=source_uri,
        source_type=SourceType.FILE if source_uri.endswith((".pdf", ".txt", ".md")) else SourceType.URL,
        content_type="text/plain",
        language="en",
        ingested_at=ingested_at,
        hash_signature=f"sha256:{random.randint(1000000, 9999999):x}",
    )
    
    return DocumentObservation(
        schema_version="1.0.0",
        metadata=metadata,
        structural=structural,
        chunk_windows=chunk_windows,
        segments=segments,
        ontology=None,
        context=context,
        observability=observability,
    )


def generate_sample_data(num_documents: int = 15) -> list[DocumentObservation]:
    """Generate a set of sample observations with varied characteristics."""
    
    observations = []
    base_time = datetime.utcnow() - timedelta(days=7)
    
    # Define document profiles with different characteristics
    profiles = [
        # Small documents
        {"token_range": (500, 2000), "chunk_range": (3, 8), "strategy": ContextStrategy.CHUNK, "risk": RiskLevel.LOW},
        # Medium documents
        {"token_range": (5000, 10000), "chunk_range": (10, 25), "strategy": ContextStrategy.WINDOW, "risk": RiskLevel.MEDIUM},
        # Large documents
        {"token_range": (15000, 25000), "chunk_range": (30, 50), "strategy": ContextStrategy.SEGMENT, "risk": RiskLevel.HIGH},
        # Very large documents
        {"token_range": (30000, 50000), "chunk_range": (50, 80), "strategy": ContextStrategy.SEGMENT, "risk": RiskLevel.HIGH},
    ]
    
    source_types = [
        "documentation/manual.pdf",
        "docs/api_reference.md",
        "https://example.com/blog/post",
        "reports/quarterly_report.pdf",
        "docs/user_guide.txt",
        "https://example.com/docs/tutorial",
    ]
    
    for i in range(num_documents):
        profile = random.choice(profiles)
        token_count = random.randint(*profile["token_range"])
        chunk_count = random.randint(*profile["chunk_range"])
        strategy = profile["strategy"]
        risk = profile["risk"]
        
        # Occasionally vary the strategy/risk
        if random.random() < 0.2:
            strategy = random.choice(list(ContextStrategy))
        if random.random() < 0.2:
            risk = random.choice(list(RiskLevel))
        
        source_uri = random.choice(source_types)
        if i < len(source_types):
            source_uri = source_types[i]
        
        ingested_at = base_time + timedelta(
            days=random.randint(0, 6),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        
        observation = generate_sample_observation(
            document_id=f"doc_{i+1:03d}",
            source_uri=source_uri,
            token_count=token_count,
            chunk_count=chunk_count,
            strategy=strategy,
            risk=risk,
            ingested_at=ingested_at,
        )
        
        observations.append(observation)
    
    return observations


def main():
    """Main demo function."""
    print("=" * 70)
    print("Spindle Analytics Dashboard Demo")
    print("=" * 70)
    print()
    
    # Create a temporary database
    db_path = Path.cwd() / "demo_analytics.db"
    if db_path.exists():
        print(f"Removing existing database at {db_path}")
        db_path.unlink()
    
    database_url = f"sqlite:///{db_path}"
    print(f"Creating analytics store at {database_url}")
    store = AnalyticsStore(database_url)
    print()
    
    # Generate sample observations
    print("Generating sample analytics observations...")
    observations = generate_sample_data(num_documents=15)
    print(f"Generated {len(observations)} observations")
    print()
    
    # Persist observations
    print("Persisting observations to analytics store...")
    store.persist_observations(observations)
    print(f"Persisted {len(observations)} observations")
    print()
    
    # Create event log store and record mocked LLM metrics
    print("Generating mocked LLM metrics...")
    event_store = EventLogStore(database_url)
    recorder = get_event_recorder()
    
    # Attach persistent observer to record events
    from spindle.observability.storage import attach_persistent_observer
    detach = attach_persistent_observer(recorder, event_store)
    
    try:
        # Generate mocked ontology recommendation events
        # Create varied distribution across scopes for better visualization
        ontology_scopes = ["minimal", "balanced", "comprehensive"]
        scope_distribution = {
            "minimal": 4,
            "balanced": 8,
            "comprehensive": 6,
        }
        
        # Model distribution for variety
        models = ["CustomFast", "CustomGPT5Mini", "CustomHaiku", "CustomOpus4"]
        
        total_ontology_calls = 0
        for scope, count in scope_distribution.items():
            for i in range(count):
                # Vary token counts based on scope
                token_ranges = {
                    "minimal": (500, 1500),
                    "balanced": (1500, 4000),
                    "comprehensive": (3000, 8000),
                }
                tokens = random.randint(*token_ranges[scope])
                # Vary cost based on model (some models are more expensive)
                model = random.choice(models)
                cost_multiplier = {
                    "CustomFast": 0.00001,
                    "CustomGPT5Mini": 0.000008,
                    "CustomHaiku": 0.000005,
                    "CustomOpus4": 0.000015,
                }.get(model, 0.00001)
                cost = tokens * cost_multiplier
                latency = random.uniform(300, 2000)
                
                ontology_recorder = get_event_recorder("ontology.recommender")
                start_time = datetime.utcnow() - timedelta(
                    days=random.randint(0, 6),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                )
                
                ontology_recorder.record(
                    "recommend.start",
                    {
                        "scope": scope,
                        "text_length": random.randint(1000, 50000),
                    },
                    timestamp=start_time,
                )
                ontology_recorder.record(
                    "recommend.complete",
                    {
                        "scope": scope,
                        "entity_type_count": {
                            "minimal": random.randint(3, 8),
                            "balanced": random.randint(6, 12),
                            "comprehensive": random.randint(10, 20),
                        }[scope],
                        "relation_type_count": {
                            "minimal": random.randint(4, 10),
                            "balanced": random.randint(8, 15),
                            "comprehensive": random.randint(12, 25),
                        }[scope],
                        "total_tokens": tokens,
                        "input_tokens": int(tokens * 0.7),
                        "output_tokens": int(tokens * 0.3),
                        "cost": cost,
                        "latency_ms": latency,
                        "model": model,
                    },
                    timestamp=start_time + timedelta(milliseconds=int(latency)),
                )
                total_ontology_calls += 1
        
        # Generate mocked triple extraction events
        # Create varied distribution across scopes
        extraction_scopes = ["minimal", "balanced", "comprehensive"]
        extraction_scope_distribution = {
            "minimal": 5,
            "balanced": 12,
            "comprehensive": 8,
        }
        
        total_extraction_calls = 0
        for scope, count in extraction_scope_distribution.items():
            for i in range(count):
                # Vary tokens based on scope
                token_ranges = {
                    "minimal": (2000, 6000),
                    "balanced": (4000, 10000),
                    "comprehensive": (6000, 15000),
                }
                tokens = random.randint(*token_ranges[scope])
                # Vary cost based on model
                model = random.choice(models)
                cost_multiplier = {
                    "CustomFast": 0.00001,
                    "CustomGPT5Mini": 0.000008,
                    "CustomHaiku": 0.000005,
                    "CustomOpus4": 0.000015,
                }.get(model, 0.00001)
                cost = tokens * cost_multiplier
                latency = random.uniform(500, 3000)
                
                # Triple count varies by scope
                triple_count = {
                    "minimal": random.randint(5, 20),
                    "balanced": random.randint(15, 40),
                    "comprehensive": random.randint(30, 60),
                }[scope]
                
                extractor_recorder = get_event_recorder("extractor")
                start_time = datetime.utcnow() - timedelta(
                    days=random.randint(0, 6),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                )
                
                extractor_recorder.record(
                    "extract.start",
                    {
                        "source_name": f"doc_{total_extraction_calls+1:03d}",
                        "source_url": None,
                        "existing_triples": random.randint(0, 10),
                    },
                    timestamp=start_time,
                )
                extractor_recorder.record(
                    "extract.complete",
                    {
                        "source_name": f"doc_{total_extraction_calls+1:03d}",
                        "triple_count": triple_count,
                        "ontology_scope": scope,
                        "total_tokens": tokens,
                        "input_tokens": int(tokens * 0.75),
                        "output_tokens": int(tokens * 0.25),
                        "cost": cost,
                        "latency_ms": latency,
                        "model": model,
                    },
                    timestamp=start_time + timedelta(milliseconds=int(latency)),
                )
                total_extraction_calls += 1
        
        # Generate mocked entity resolution events
        # Create more varied data for entity and edge matching
        total_entity_matching = 0
        total_edge_matching = 0
        
        # Entity matching events
        for i in range(8):
            tokens = random.randint(1000, 5000)
            model = random.choice(models)
            cost_multiplier = {
                "CustomFast": 0.00001,
                "CustomGPT5Mini": 0.000008,
                "CustomHaiku": 0.000005,
                "CustomOpus4": 0.000015,
            }.get(model, 0.00001)
            cost = tokens * cost_multiplier
            latency = random.uniform(200, 1500)
            block_size = random.randint(10, 50)
            
            resolution_recorder = get_event_recorder("entity_resolution")
            start_time = datetime.utcnow() - timedelta(
                days=random.randint(0, 6),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            
            resolution_recorder.record(
                "matching.entities.start",
                {
                    "block_size": block_size,
                    "has_context": random.choice([True, False]),
                },
                timestamp=start_time,
            )
            resolution_recorder.record(
                "matching.entities.complete",
                {
                    "block_size": block_size,
                    "matches_found": random.randint(2, min(20, block_size // 2)),
                    "total_tokens": tokens,
                    "cost": cost,
                    "latency_ms": latency,
                    "model": model,
                },
                timestamp=start_time + timedelta(milliseconds=int(latency)),
            )
            total_entity_matching += 1
        
        # Edge matching events
        for i in range(7):
            tokens = random.randint(800, 4000)
            model = random.choice(models)
            cost_multiplier = {
                "CustomFast": 0.00001,
                "CustomGPT5Mini": 0.000008,
                "CustomHaiku": 0.000005,
                "CustomOpus4": 0.000015,
            }.get(model, 0.00001)
            cost = tokens * cost_multiplier
            latency = random.uniform(150, 1200)
            block_size = random.randint(8, 40)
            
            resolution_recorder = get_event_recorder("entity_resolution")
            start_time = datetime.utcnow() - timedelta(
                days=random.randint(0, 6),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            
            resolution_recorder.record(
                "matching.edges.start",
                {
                    "block_size": block_size,
                    "has_context": random.choice([True, False]),
                },
                timestamp=start_time,
            )
            resolution_recorder.record(
                "matching.edges.complete",
                {
                    "block_size": block_size,
                    "matches_found": random.randint(1, min(15, block_size // 2)),
                    "total_tokens": tokens,
                    "cost": cost,
                    "latency_ms": latency,
                    "model": model,
                },
                timestamp=start_time + timedelta(milliseconds=int(latency)),
            )
            total_edge_matching += 1
        
        print("Generated mocked LLM metrics for:")
        print(f"  - {total_ontology_calls} ontology recommendation calls")
        print(f"    ({scope_distribution['minimal']} minimal, {scope_distribution['balanced']} balanced, {scope_distribution['comprehensive']} comprehensive)")
        print(f"  - {total_extraction_calls} triple extraction calls")
        print(f"    ({extraction_scope_distribution['minimal']} minimal, {extraction_scope_distribution['balanced']} balanced, {extraction_scope_distribution['comprehensive']} comprehensive)")
        print(f"  - {total_entity_matching + total_edge_matching} entity resolution operations")
        print(f"    ({total_entity_matching} entity matching, {total_edge_matching} edge matching)")
        print()
    finally:
        detach()
    
    # Display summary
    print("Sample Data Summary:")
    print("-" * 70)
    total_tokens = sum(obs.structural.token_count for obs in observations)
    total_chunks = sum(obs.structural.chunk_count for obs in observations)
    strategies = {}
    risks = {}
    for obs in observations:
        if obs.context:
            strategy = obs.context.recommended_strategy.value
            strategies[strategy] = strategies.get(strategy, 0) + 1
            risk = obs.context.supporting_risk.value
            risks[risk] = risks.get(risk, 0) + 1
    
    print(f"Total Documents: {len(observations)}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Total Chunks: {total_chunks:,}")
    print(f"Average Tokens per Document: {total_tokens / len(observations):.1f}")
    print(f"Average Chunks per Document: {total_chunks / len(observations):.1f}")
    print()
    print("Context Strategy Distribution:")
    for strategy, count in sorted(strategies.items()):
        print(f"  {strategy}: {count}")
    print()
    print("Risk Level Distribution:")
    for risk, count in sorted(risks.items()):
        print(f"  {risk}: {count}")
    print()
    
    # Launch Streamlit dashboard
    print("=" * 70)
    print("Launching Streamlit Dashboard...")
    print("=" * 70)
    print()
    print(f"Database location: {db_path}")
    print()
    print("The dashboard will open in your browser automatically.")
    print("Press Ctrl+C to stop the dashboard server.")
    print()
    
    # Launch Streamlit
    try:
        # Set the database URL as an environment variable for the dashboard to pick up
        env = os.environ.copy()
        env["SPINDLE_DATABASE"] = database_url
        
        # Launch streamlit
        dashboard_path = Path(__file__).parent.parent / "spindle" / "dashboard" / "app.py"
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            "8501",
            "--server.headless",
            "false",
        ]
        
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user.")
    except Exception as e:
        print(f"\n\nError launching dashboard: {e}", file=sys.stderr)
        print("\nYou can manually launch the dashboard with:")
        print(f"  SPINDLE_DATABASE={database_url} streamlit run spindle/dashboard/app.py")
        print("Or use the CLI:")
        print(f"  python -m spindle.dashboard.app --database {db_path}")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

