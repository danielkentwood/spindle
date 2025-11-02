# Conda to UV Migration Quick Reference

This guide helps you migrate from conda to uv for the Spindle project.

## Why Migrate?

- ⚡ **10-100x faster** installation
- 🤖 **AI-Friendly**: Clearer commands that AI agents remember better
- 🔒 **Better dependency resolution**
- 💾 **Smaller footprint**: No heavy conda base environment

## Quick Migration Steps

### 1. Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then restart terminal or:
source $HOME/.cargo/env
```

### 2. Setup New Environment

```bash
cd /path/to/spindle

# Create venv with uv
uv venv

# Install all dependencies
uv pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Check it works
uv run pytest tests/ -m "not integration"
```

### 4. Optional: Deactivate Conda

If you had conda auto-activation in your shell:

```bash
# Remove from ~/.bashrc or ~/.zshrc
conda config --set auto_activate_base false
```

## Command Translation

| Conda Command | UV Equivalent |
|--------------|---------------|
| `conda activate kgx` | *Not needed - use `uv run`* |
| `conda install package` | `uv pip install package` |
| `pip install package` | `uv pip install package` |
| `python script.py` | `uv run python script.py` |
| `pytest` | `uv run pytest` |
| `pip install -e .` | `uv pip install -e .` |
| `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| `conda list` | `uv pip list` |
| `conda env export` | `uv pip freeze > requirements.txt` |

## Key Differences

### With Conda (Old Way)
```bash
# Had to remember to activate
conda activate kgx

# Then run commands
python demos/example.py
pytest
```

### With UV (New Way)
```bash
# No activation needed - just prefix with 'uv run'
uv run python demos/example.py
uv run pytest

# The agent will remember this pattern better!
```

## What Changed in the Project?

✅ **Added Files:**
- `pyproject.toml` - Modern Python project config
- `.python-version` - Specifies Python 3.11
- `docs/UV_SETUP.md` - Comprehensive uv guide
- `.cursor/rules/uv.mdc` - AI agent instructions

✅ **Updated Files:**
- `README.md` - Updated installation instructions
- `.gitignore` - Added `.venv/` and uv artifacts

✅ **Removed:**
- `.cursor/rules/conda.mdc` - Replaced with uv.mdc

## Troubleshooting

### "Command not found: uv"

```bash
# Reinstall or add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

### "Python version not found"

```bash
# Install Python 3.11
brew install python@3.11

# Then recreate environment
rm -rf .venv
uv venv
```

### Import errors

```bash
# Reinstall in editable mode
uv pip install -e ".[dev]"
```

### Still want to use conda?

That's fine! The project still works with conda. Just use:
```bash
conda activate kgx
pip install -e ".[dev]"
python script.py  # without 'uv run'
```

But update `.cursor/rules/uv.mdc` back to conda instructions.

## Benefits for AI Agents

The main reason for switching to uv is that AI agents work better with it:

1. **No activation needed**: `uv run` handles everything
2. **Clearer pattern**: Always prefix with `uv run`
3. **Less context**: No need to track "am I in the environment?"
4. **Explicit commands**: Every command is self-contained

With conda, agents often forgot to activate the environment, leading to failures.
With uv, the pattern is simple and consistent.

## Next Steps

1. ✅ Install uv
2. ✅ Run `uv venv` and `uv pip install -e ".[dev]"`
3. ✅ Test with `uv run pytest`
4. 📚 Read [docs/UV_SETUP.md](UV_SETUP.md) for more details
5. 🚀 Start using `uv run` for all Python commands

## Questions?

See the full guide: [docs/UV_SETUP.md](UV_SETUP.md)

