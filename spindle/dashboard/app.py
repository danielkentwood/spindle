"""Streamlit analytics dashboard for Spindle observations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import pandas as pd

from spindle.analytics import (
    AnalyticsStore,
    chunk_window_risk,
    corpus_overview,
    document_size_table,
    entity_resolution_metrics,
    ontology_recommendation_metrics,
    triple_extraction_metrics,
)


def _normalize_database(value: str) -> str:
    """Normalize database URL or path to SQLite URL format."""
    if value.startswith("sqlite://"):
        return value
    path = Path(value).expanduser().resolve()
    return f"sqlite:///{path}"


@st.cache_data
def load_data(database_url: str, limit: int | None = None) -> dict[str, Any]:
    """Load analytics data from the store."""
    store = AnalyticsStore(database_url)
    return {
        "overview": corpus_overview(store, limit=limit),
        "documents": document_size_table(store, limit=limit),
        "windows": chunk_window_risk(store, limit=limit),
        "ontology_metrics": ontology_recommendation_metrics(store, limit=limit),
        "extraction_metrics": triple_extraction_metrics(store, limit=limit),
        "resolution_metrics": entity_resolution_metrics(store, limit=limit),
    }


def render_document_ingestion_tab(data: dict[str, Any]) -> None:
    """Render the Document Ingestion tab."""
    overview = data["overview"]
    documents = data["documents"]
    windows = data["windows"]
    
    st.header("Corpus Overview")
    st.caption(
        "High-level statistics about your document corpus, including aggregate token counts, "
        "chunking patterns, and recommended processing strategies."
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", overview["documents"])
        st.caption("The total number of documents that have been ingested and analyzed.")
    with col2:
        st.metric("Total Tokens", f"{overview['total_tokens']:,}")
        st.caption(
            "Sum of all tokens across all documents. Tokens are the basic units used by language models "
            "(approximately 4 characters or 0.75 words per token)."
        )
    with col3:
        st.metric("Avg Tokens/Doc", f"{overview['avg_tokens']:.1f}")
        st.caption(
            "Mean token count across all documents. Helps identify if your corpus contains mostly "
            "small, medium, or large documents."
        )
    with col4:
        st.metric("Avg Chunks/Doc", f"{overview['avg_chunks']:.1f}")
        st.caption(
            "Mean number of chunks created per document. Documents are split into chunks to fit within "
            "language model context windows and improve processing efficiency."
        )
    
    st.subheader("Context Strategy Distribution")
    with st.expander("What are context strategies?"):
        st.markdown("""
        Shows how many documents are recommended for each processing strategy:
        
        - **chunk**: Process individual chunks (for very large documents)
        - **window**: Process sliding windows of chunks (for medium-large documents)
        - **segment**: Process semantic segments (for documents with clear topic boundaries)
        - **document**: Process entire document at once (for small documents that fit in context)
        """)
    if overview["context_strategy_counts"]:
        strategy_df = pd.DataFrame(
            list(overview["context_strategy_counts"].items()),
            columns=["Strategy", "Count"]
        )
        st.bar_chart(strategy_df.set_index("Strategy"))
    else:
        st.info("No strategy data available.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Document Sizes")
        st.caption(
            "Per-document metrics showing token counts, chunking structure, and processing recommendations."
        )
        if documents:
            doc_df = pd.DataFrame(documents)
            st.dataframe(
                doc_df[["document_id", "token_count", "chunk_count", "context_strategy", "risk_level"]],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Column Descriptions"):
                st.markdown("""
                - **Document ID**: Unique identifier for each document
                - **Tokens**: Total token count for the document
                - **Chunks**: Number of chunks the document was split into
                - **Strategy**: Recommended processing strategy (chunk/window/segment/document)
                - **Risk**: Overall risk level (low/medium/high) based on token usage relative to context budget
                """)
        else:
            st.info("No document data available.")
    
    with col2:
        st.subheader("Chunk Window Risk")
        st.caption(
            "Analysis of sliding windows of consecutive chunks to identify potential context limit violations. "
            "Windows help maintain semantic context across chunk boundaries but may exceed token budgets."
        )
        if windows:
            window_df = pd.DataFrame(windows)
            st.dataframe(
                window_df[["document_id", "window_size", "max_tokens", "median_tokens", "risk"]],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Column Descriptions"):
                st.markdown("""
                - **Document**: Document identifier
                - **Window Size**: Number of consecutive chunks in the sliding window (e.g., 2, 3, or 5 chunks)
                - **Max Tokens**: Maximum token count across all windows of this size
                - **Median Tokens**: Median token count across all windows of this size
                - **Risk**: Risk level indicating likelihood of exceeding context budget:
                  - **Low**: Max tokens ≤ 60% of budget (≤7,200 tokens)
                  - **Medium**: Max tokens 60-110% of budget (7,200-13,200 tokens)
                  - **High**: Max tokens > 110% of budget (>13,200 tokens)
                """)
            with st.expander("What is chunk window risk?"):
                st.markdown("""
                When processing documents, you may want to send multiple consecutive chunks together to maintain context. 
                However, if a window exceeds your token budget (typically 12,000 tokens), it may be truncated or cause 
                processing failures. This table helps identify which documents and window sizes are at risk.
                """)
        else:
            st.info("No window risk data available.")


def render_ontology_recommendation_tab(data: dict[str, Any]) -> None:
    """Render the Ontology Recommendation tab."""
    metrics = data["ontology_metrics"]
    
    st.header("Ontology Recommendation Metrics")
    st.caption("LLM usage metrics for ontology recommendation operations, grouped by scope.")
    
    if metrics["total_calls"] == 0:
        st.info("No ontology recommendation metrics available yet.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Calls", metrics["total_calls"])
    with col2:
        st.metric("Total Tokens", f"{metrics['total_tokens']:,}")
    with col3:
        st.metric("Total Cost", f"${metrics['total_cost']:.4f}")
    with col4:
        st.metric("Avg Latency", f"{metrics['avg_latency_ms']:.1f} ms")
    
    st.divider()
    
    if metrics["by_scope"]:
        st.subheader("Metrics by Scope")
        scope_data = []
        for scope, scope_metrics in metrics["by_scope"].items():
            scope_data.append({
                "Scope": scope,
                "Calls": scope_metrics["calls"],
                "Total Tokens": scope_metrics["total_tokens"],
                "Input Tokens": scope_metrics.get("input_tokens", "N/A"),
                "Output Tokens": scope_metrics.get("output_tokens", "N/A"),
                "Cost": f"${scope_metrics['total_cost']:.4f}",
                "Avg Latency (ms)": f"{scope_metrics['avg_latency_ms']:.1f}",
            })
        
        scope_df = pd.DataFrame(scope_data)
        st.dataframe(scope_df, use_container_width=True, hide_index=True)
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Token Usage by Scope")
            token_chart_df = pd.DataFrame([
                {"Scope": scope, "Tokens": m["total_tokens"]}
                for scope, m in metrics["by_scope"].items()
            ])
            st.bar_chart(token_chart_df.set_index("Scope"))
        
        with col2:
            st.subheader("Cost Breakdown by Scope")
            cost_chart_df = pd.DataFrame([
                {"Scope": scope, "Cost": m["total_cost"]}
                for scope, m in metrics["by_scope"].items()
            ])
            st.bar_chart(cost_chart_df.set_index("Scope"))
    else:
        st.info("No scope-specific data available.")
    
    # Model breakdown
    if metrics.get("by_model"):
        st.divider()
        st.subheader("Metrics by Model")
        model_data = []
        for model, model_metrics in metrics["by_model"].items():
            model_data.append({
                "Model": model,
                "Calls": model_metrics["calls"],
                "Total Tokens": model_metrics["total_tokens"],
                "Input Tokens": model_metrics.get("input_tokens", "N/A"),
                "Output Tokens": model_metrics.get("output_tokens", "N/A"),
                "Cost": f"${model_metrics['total_cost']:.4f}",
                "Avg Latency (ms)": f"{model_metrics['avg_latency_ms']:.1f}",
            })
        
        model_df = pd.DataFrame(model_data)
        st.dataframe(model_df, use_container_width=True, hide_index=True)
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Token Usage by Model")
            token_chart_df = pd.DataFrame([
                {"Model": model, "Tokens": m["total_tokens"]}
                for model, m in metrics["by_model"].items()
            ])
            st.bar_chart(token_chart_df.set_index("Model"))
        
        with col2:
            st.subheader("Cost Breakdown by Model")
            cost_chart_df = pd.DataFrame([
                {"Model": model, "Cost": m["total_cost"]}
                for model, m in metrics["by_model"].items()
            ])
            st.bar_chart(cost_chart_df.set_index("Model"))


def render_triple_extraction_tab(data: dict[str, Any]) -> None:
    """Render the Triple Extraction tab."""
    metrics = data["extraction_metrics"]
    
    st.header("Triple Extraction Metrics")
    st.caption(
        "LLM usage metrics for triple extraction operations, grouped by ontology scope (strategy). "
        "Also includes entity and triple counts normalized against document size."
    )
    
    if metrics["total_calls"] == 0:
        st.info("No triple extraction metrics available yet.")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Calls", metrics["total_calls"])
    with col2:
        st.metric("Total Tokens", f"{metrics['total_tokens']:,}")
    with col3:
        st.metric("Total Cost", f"${metrics['total_cost']:.4f}")
    with col4:
        st.metric("Avg Latency", f"{metrics['avg_latency_ms']:.1f} ms")
    with col5:
        st.metric("Total Triples", metrics["total_triples"])
    
    st.divider()
    
    if metrics["by_scope"]:
        st.subheader("Metrics by Ontology Scope (Strategy)")
        scope_data = []
        for scope, scope_metrics in metrics["by_scope"].items():
            scope_data.append({
                "Scope": scope,
                "Calls": scope_metrics["calls"],
                "Total Tokens": scope_metrics["total_tokens"],
                "Input Tokens": scope_metrics.get("input_tokens", "N/A"),
                "Output Tokens": scope_metrics.get("output_tokens", "N/A"),
                "Cost": f"${scope_metrics['total_cost']:.4f}",
                "Avg Latency (ms)": f"{scope_metrics['avg_latency_ms']:.1f}",
                "Triples": scope_metrics["triples"],
                "Avg Triples/Call": f"{scope_metrics.get('avg_triples_per_call', 0):.1f}",
            })
        
        scope_df = pd.DataFrame(scope_data)
        st.dataframe(scope_df, use_container_width=True, hide_index=True)
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Token Usage by Scope")
            token_chart_df = pd.DataFrame([
                {"Scope": scope, "Tokens": m["total_tokens"]}
                for scope, m in metrics["by_scope"].items()
            ])
            st.bar_chart(token_chart_df.set_index("Scope"))
        
        with col2:
            st.subheader("Triples Extracted by Scope")
            triples_chart_df = pd.DataFrame([
                {"Scope": scope, "Triples": m["triples"]}
                for scope, m in metrics["by_scope"].items()
            ])
            st.bar_chart(triples_chart_df.set_index("Scope"))
        
        # Efficiency metrics
        st.subheader("Extraction Efficiency")
        efficiency_data = []
        for scope, scope_metrics in metrics["by_scope"].items():
            if scope_metrics["total_tokens"] > 0:
                triples_per_token = scope_metrics["triples"] / scope_metrics["total_tokens"]
                triples_per_cost = scope_metrics["triples"] / scope_metrics["total_cost"] if scope_metrics["total_cost"] > 0 else 0
            else:
                triples_per_token = 0
                triples_per_cost = 0
            
            efficiency_data.append({
                "Scope": scope,
                "Triples per Token": f"{triples_per_token:.4f}",
                "Triples per $": f"{triples_per_cost:.2f}",
            })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        st.dataframe(efficiency_df, use_container_width=True, hide_index=True)
    else:
        st.info("No scope-specific data available.")
    
    # Model breakdown
    if metrics.get("by_model"):
        st.divider()
        st.subheader("Metrics by Model")
        model_data = []
        for model, model_metrics in metrics["by_model"].items():
            model_data.append({
                "Model": model,
                "Calls": model_metrics["calls"],
                "Total Tokens": model_metrics["total_tokens"],
                "Input Tokens": model_metrics.get("input_tokens", "N/A"),
                "Output Tokens": model_metrics.get("output_tokens", "N/A"),
                "Cost": f"${model_metrics['total_cost']:.4f}",
                "Avg Latency (ms)": f"{model_metrics['avg_latency_ms']:.1f}",
                "Triples": model_metrics["triples"],
                "Avg Triples/Call": f"{model_metrics.get('avg_triples_per_call', 0):.1f}",
            })
        
        model_df = pd.DataFrame(model_data)
        st.dataframe(model_df, use_container_width=True, hide_index=True)
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Token Usage by Model")
            token_chart_df = pd.DataFrame([
                {"Model": model, "Tokens": m["total_tokens"]}
                for model, m in metrics["by_model"].items()
            ])
            st.bar_chart(token_chart_df.set_index("Model"))
        
        with col2:
            st.subheader("Cost Breakdown by Model")
            cost_chart_df = pd.DataFrame([
                {"Model": model, "Cost": m["total_cost"]}
                for model, m in metrics["by_model"].items()
            ])
            st.bar_chart(cost_chart_df.set_index("Model"))


def render_entity_resolution_tab(data: dict[str, Any]) -> None:
    """Render the Entity Resolution tab."""
    metrics = data["resolution_metrics"]
    
    st.header("Entity Resolution Metrics")
    st.caption("LLM usage metrics for entity resolution operations, broken down by resolution step.")
    
    if metrics["total_calls"] == 0:
        st.info("No entity resolution metrics available yet.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Calls", metrics["total_calls"])
    with col2:
        st.metric("Total Tokens", f"{metrics['total_tokens']:,}")
    with col3:
        st.metric("Total Cost", f"${metrics['total_cost']:.4f}")
    with col4:
        st.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.1f} ms")
    
    st.divider()
    
    st.subheader("Metrics by Resolution Step")
    
    step_data = []
    for step_name, step_metrics in [
        ("Entity Matching", metrics["entity_matching"]),
        ("Edge Matching", metrics["edge_matching"]),
    ]:
        step_data.append({
            "Step": step_name,
            "Calls": step_metrics["calls"],
            "Total Tokens": step_metrics["total_tokens"],
            "Cost": f"${step_metrics['total_cost']:.4f}",
            "Avg Latency (ms)": f"{step_metrics['avg_latency_ms']:.1f}",
        })
    
    step_df = pd.DataFrame(step_data)
    st.dataframe(step_df, use_container_width=True, hide_index=True)
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Token Usage by Step")
        token_chart_df = pd.DataFrame([
            {"Step": "Entity Matching", "Tokens": metrics["entity_matching"]["total_tokens"]},
            {"Step": "Edge Matching", "Tokens": metrics["edge_matching"]["total_tokens"]},
        ])
        st.bar_chart(token_chart_df.set_index("Step"))
    
    with col2:
        st.subheader("Cost Breakdown by Step")
        cost_chart_df = pd.DataFrame([
            {"Step": "Entity Matching", "Cost": metrics["entity_matching"]["total_cost"]},
            {"Step": "Edge Matching", "Cost": metrics["edge_matching"]["total_cost"]},
        ])
        st.bar_chart(cost_chart_df.set_index("Step"))
    
    # Model breakdowns
    if metrics["entity_matching"].get("by_model"):
        st.divider()
        st.subheader("Entity Matching by Model")
        entity_model_data = []
        for model, model_metrics in metrics["entity_matching"]["by_model"].items():
            entity_model_data.append({
                "Model": model,
                "Calls": model_metrics["calls"],
                "Total Tokens": model_metrics["total_tokens"],
                "Cost": f"${model_metrics['total_cost']:.4f}",
                "Avg Latency (ms)": f"{model_metrics['avg_latency_ms']:.1f}",
            })
        
        entity_model_df = pd.DataFrame(entity_model_data)
        st.dataframe(entity_model_df, use_container_width=True, hide_index=True)
    
    if metrics["edge_matching"].get("by_model"):
        st.divider()
        st.subheader("Edge Matching by Model")
        edge_model_data = []
        for model, model_metrics in metrics["edge_matching"]["by_model"].items():
            edge_model_data.append({
                "Model": model,
                "Calls": model_metrics["calls"],
                "Total Tokens": model_metrics["total_tokens"],
                "Cost": f"${model_metrics['total_cost']:.4f}",
                "Avg Latency (ms)": f"{model_metrics['avg_latency_ms']:.1f}",
            })
        
        edge_model_df = pd.DataFrame(edge_model_data)
        st.dataframe(edge_model_df, use_container_width=True, hide_index=True)


def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="Spindle Analytics Dashboard",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("📊 Spindle Analytics Dashboard")
    
    # Get database URL from query params or sidebar
    database_url = st.query_params.get("database", "")
    if not database_url:
        # Try environment variable as fallback
        import os
        database_url = os.environ.get("SPINDLE_DATABASE", "")
    
    if not database_url:
        database_url = st.sidebar.text_input(
            "Database URL or Path",
            value="",
            help="SQLite database URL (e.g., sqlite:///analytics.db) or file path",
        )
    
    limit = st.sidebar.number_input(
        "Limit Observations",
        min_value=None,
        max_value=None,
        value=None,
        help="Optional limit on the number of observations to process",
    )
    
    if not database_url:
        st.warning("Please provide a database URL or path in the sidebar.")
        st.stop()
    
    try:
        database_url = _normalize_database(database_url)
        data = load_data(database_url, limit=limit)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Document Ingestion",
        "🔍 Ontology Recommendation",
        "🔗 Triple Extraction",
        "🔀 Entity Resolution",
    ])
    
    with tab1:
        render_document_ingestion_tab(data)
    
    with tab2:
        render_ontology_recommendation_tab(data)
    
    with tab3:
        render_triple_extraction_tab(data)
    
    with tab4:
        render_entity_resolution_tab(data)


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point for running the Streamlit app."""
    parser = argparse.ArgumentParser(description="Launch Spindle analytics dashboard.")
    parser.add_argument(
        "--database",
        required=True,
        help="SQLite database URL or path containing analytics observations.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501).",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to run Streamlit on (default: localhost).",
    )
    args = parser.parse_args(argv)
    
    database_url = _normalize_database(args.database)
    
    # Run streamlit with query parameter
    import subprocess
    
    # Use streamlit's --server.runOnSave to auto-reload
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        __file__,
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        "--",
        f"?database={database_url}",
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        return 0
    except subprocess.CalledProcessError as e:
        return e.returncode
    
    return 0


# Streamlit executes the script directly, so main() will be called
# For CLI usage with --database, handle separately
if __name__ == "__main__":
    # Check if this is a direct CLI invocation (not via streamlit)
    if len(sys.argv) > 1 and "--database" in sys.argv:
        # Check if streamlit is in the command
        cmd_str = " ".join(sys.argv).lower()
        if "streamlit" not in cmd_str and "run" not in cmd_str:
            raise SystemExit(cli_main(sys.argv[1:]))
    
    # Otherwise, streamlit will execute main() when it runs the script
    # We call it here for direct execution
    try:
        main()
    except Exception:
        # If main() fails (e.g., streamlit not available), try CLI
        if "--database" in sys.argv:
            raise SystemExit(cli_main(sys.argv[1:]))
        raise
