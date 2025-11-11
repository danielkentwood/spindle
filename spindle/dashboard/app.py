"""Generate a lightweight HTML dashboard for analytics observations."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any

from spindle.analytics import (
    AnalyticsStore,
    chunk_window_risk,
    corpus_overview,
    document_size_table,
)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Spindle Analytics Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 2rem;
      background-color: #f6f7fb;
      color: #1f2933;
    }}
    h1, h2 {{
      color: #0b7285;
    }}
    .card {{
      background: #ffffff;
      border-radius: 8px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
    }}
    th, td {{
      padding: 0.75rem;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      font-size: 0.95rem;
    }}
    th {{
      background-color: #eff6ff;
      color: #1e3a8a;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    tr:hover {{
      background-color: #f1f5f9;
    }}
    canvas {{
      max-width: 600px;
      margin-top: 1rem;
    }}
    .grid {{
      display: grid;
      gap: 1.5rem;
    }}
    @media (min-width: 960px) {{
      .grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <h1>Spindle Analytics Dashboard</h1>
  <section class="card">
    <h2>Corpus Overview</h2>
    <p>Total Documents: <strong id="doc-count"></strong></p>
    <p>Total Tokens: <strong id="total-tokens"></strong></p>
    <p>Average Tokens per Document: <strong id="avg-tokens"></strong></p>
    <p>Average Chunks per Document: <strong id="avg-chunks"></strong></p>
    <canvas id="strategyChart"></canvas>
  </section>

  <section class="grid">
    <div class="card">
      <h2>Document Sizes</h2>
      <table id="document-table">
        <thead>
          <tr>
            <th>Document ID</th>
            <th>Tokens</th>
            <th>Chunks</th>
            <th>Strategy</th>
            <th>Risk</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <h2>Chunk Window Risk</h2>
      <table id="window-table">
        <thead>
          <tr>
            <th>Document</th>
            <th>Window Size</th>
            <th>Max Tokens</th>
            <th>Median Tokens</th>
            <th>Risk</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <script id="dashboard-data" type="application/json">{data}</script>
  <script>
    const payload = JSON.parse(document.getElementById('dashboard-data').textContent);

    function formatNumber(value) {{
      return new Intl.NumberFormat().format(value);
    }}

    function populateOverview() {{
      const overview = payload.overview;
      document.getElementById('doc-count').textContent = overview.documents;
      document.getElementById('total-tokens').textContent = formatNumber(overview.total_tokens);
      document.getElementById('avg-tokens').textContent = overview.avg_tokens.toFixed(1);
      document.getElementById('avg-chunks').textContent = overview.avg_chunks.toFixed(1);

      const ctx = document.getElementById('strategyChart');
      const strategies = Object.keys(overview.context_strategy_counts);
      const counts = Object.values(overview.context_strategy_counts);
      new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: strategies,
          datasets: [{{
            label: 'Context Strategy Count',
            data: counts,
            backgroundColor: 'rgba(14, 116, 144, 0.6)',
            borderColor: 'rgba(14, 116, 144, 1)',
            borderWidth: 1,
          }}],
        }},
        options: {{
          scales: {{
            y: {{
              beginAtZero: true,
              ticks: {{
                precision: 0
              }}
            }}
          }}
        }},
      }});
    }}

    function populateTable(tableId, rows, columns) {{
      const body = document.querySelector(`${{tableId}} tbody`);
      body.innerHTML = '';
      rows.forEach(row => {{
        const tr = document.createElement('tr');
        columns.forEach(column => {{
          const td = document.createElement('td');
          let value = row[column] ?? '';
          if (typeof value === 'number') {{
            value = formatNumber(value);
          }}
          td.textContent = value;
          tr.appendChild(td);
        }});
        body.appendChild(tr);
      }});
    }}

    populateOverview();
    populateTable('#document-table', payload.documents, ['document_id', 'token_count', 'chunk_count', 'context_strategy', 'risk_level']);
    populateTable('#window-table', payload.windows, ['document_id', 'window_size', 'max_tokens', 'median_tokens', 'risk']);
  </script>
</body>
</html>
"""


def build_dashboard(
    database_url: str,
    *,
    output_path: Path | None = None,
    limit: int | None = None,
) -> Path:
    """Render the dashboard HTML file and return its path."""

    store = AnalyticsStore(database_url)
    overview = corpus_overview(store, limit=limit)
    documents = document_size_table(store, limit=limit)
    windows = chunk_window_risk(store, limit=limit)

    data = {
        "overview": overview,
        "documents": documents,
        "windows": windows,
    }

    destination = output_path or Path.cwd() / "analytics_dashboard.html"
    destination.write_text(HTML_TEMPLATE.format(data=json.dumps(data)), encoding="utf-8")
    return destination


def _normalize_database(value: str) -> str:
    if value.startswith("sqlite://"):
        return value
    path = Path(value).expanduser().resolve()
    return f"sqlite:///{path}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a static analytics dashboard.")
    parser.add_argument(
        "--database",
        required=True,
        help="SQLite database URL or path containing analytics observations.",
    )
    parser.add_argument(
        "--output",
        help="Path where the dashboard HTML should be written. Defaults to ./analytics_dashboard.html",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on the number of observations processed.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated dashboard in the default browser.",
    )
    args = parser.parse_args(argv)

    database_url = _normalize_database(args.database)
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    destination = build_dashboard(
        database_url,
        output_path=output_path,
        limit=args.limit,
    )
    if args.open:
        webbrowser.open(destination.as_uri())
    print(f"Dashboard written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

