# Roadmap

## Core Features

- [x] Add support for more LLM providers
- [ ] Add support for more graph storage providers and query languages
- [ ] Add support for graph visualization
- [x] Add dedicated module for semantic entity resolution
- [ ] Add support for cross-modal retrieval (graph + text)

## API

- [ ] Research API vs MCP
- [ ] Research pros and cons of unified API vs separate APIs for different services

## Document graph

- Document graph is a graph that contains documents and their chunks. - each document has a node in the graph with the document name as the node name. It has the following properties:
- type: "document"
- metadata:
  - doc_store_name: the name of the document store
  - doc_store_id: the id of the document store
  - summary: the summary of the document- chunks are children of the document node. They have the following properties:
- type: "chunk"
- metadata:

## Observability

- [ ] Add basic dashboard for monitoring and analysis of logs, metrics, and events