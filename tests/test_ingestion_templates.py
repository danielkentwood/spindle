from pathlib import Path

from spindle.ingestion.templates import load_templates_from_paths


def test_load_templates_from_yaml(tmp_path: Path) -> None:
    template_path = tmp_path / "templates.yaml"
    template_path.write_text(
        """
name: dummy
selector:
  extensions: [".txt"]
loader: spindle.tests.fixtures.ingestion.DummyLoader
splitter:
  name: spindle.tests.fixtures.ingestion.DummySplitter
""",
        encoding="utf-8",
    )

    specs = load_templates_from_paths([template_path])

    assert specs[0].name == "dummy"
    assert ".txt" in specs[0].selector.file_extensions
    assert specs[0].loader == "spindle.tests.fixtures.ingestion.DummyLoader"

