# Frequently Asked Questions

## How does chunking work?
Spindle relies on LangChain splitters configured per template. You can
override chunk sizes, separators, or even switch to semantic splitters.

## Where are ingestion metrics stored?
Metrics are returned with every run. Optionally, connect a document catalog
or vector store to persist them.

## Can I add custom metadata?
Yes! Provide metadata extractor callables in your template to populate
document-level context (titles, tags, owners, etc.).
