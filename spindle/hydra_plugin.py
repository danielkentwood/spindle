"""Hydra SearchPathPlugin for spindle config composition.

When spindle is installed, this plugin injects spindle's config directory
into Hydra's search path, making spindle-specific config groups available
for composition with spindle-eval's evaluation configs.

Registered via the entry point in pyproject.toml:

    [project.entry-points."hydra.searchpath"]
    spindle = "spindle.hydra_plugin:SpindleSearchPathPlugin"
"""

from __future__ import annotations


class SpindleSearchPathPlugin:
    """Hydra SearchPathPlugin that exposes spindle's conf/ directory.

    The import of ``hydra`` types is deferred so that spindle can be imported
    without ``hydra-core`` installed.  This plugin is only active when Hydra
    discovers it through the ``hydra.searchpath`` entry point.
    """

    def manipulate_search_path(self, search_path: object) -> None:  # type: ignore[override]
        search_path.append(  # type: ignore[attr-defined]
            provider="spindle",
            path="pkg://spindle.conf",
        )


def _register_plugin() -> None:
    """Conditionally subclass SearchPathPlugin when hydra-core is available."""
    try:
        from hydra.core.config_search_path import ConfigSearchPath
        from hydra.plugins.search_path_plugin import SearchPathPlugin

        class _HydraSpindleSearchPathPlugin(SearchPathPlugin):  # type: ignore[misc]
            def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
                search_path.append(
                    provider="spindle",
                    path="pkg://spindle.conf",
                )

        # Overwrite the stub with the real subclass
        global SpindleSearchPathPlugin
        SpindleSearchPathPlugin = _HydraSpindleSearchPathPlugin  # type: ignore[misc]

    except ImportError:
        pass


_register_plugin()
