# UV Setup Guide

This project uses [uv](https://github.com/astral-sh/uv) for Python package and environment management. UV is a fast, reliable alternative to pip and conda written in Rust.

## Why UV?

- ⚡ **10-100x faster** than pip/conda
- 🔒 **Reliable**: Better dependency resolution
- 🎯 **Simple**: Drop-in replacement for pip
- 🤖 **AI-Friendly**: Clear commands that AI agents remember
- 📦 **Modern**: Works great with pyproject.toml

## Installation

### Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with Homebrew
brew install uv
```

After installation, restart your terminal or run:
```bash
source $HOME/.cargo/env
```

Verify installation:
```bash
uv --version
```

## Quick Start

### 1. Create Virtual Environment

```bash
# Navigate to project root
cd /path/to/spindle

# Create virtual environment (uses Python from .python-version)
uv venv
```

This creates a `.venv/` directory with your virtual environment.

### 2. Install Dependencies

```bash
# Install all dependencies (production + dev)
uv pip install -e ".[dev]"

# Or install production only
uv pip install -e .

# Or install from requirements files
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

The `-e` flag installs in "editable" mode, so your code changes are immediately available.

### 3. Run Commands

```bash
# Run Python scripts
uv run python demos/example.py

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=spindle --cov-report=html

# Run any Python command
uv run python -c "import spindle; print(spindle.__version__)"
```

## Daily Workflow

### Running Scripts
```bash
# Always prefix with 'uv run'
uv run python your_script.py
```

### Adding New Dependencies

```bash
# Install the package
uv pip install some-package

# Then add to pyproject.toml under [project.dependencies]
# Or add to requirements.txt for pip compatibility
```

### Development Packages

```bash
# Install a dev dependency
uv pip install pytest-asyncio

# Add to pyproject.toml under [project.optional-dependencies.dev]
```

### Updating Dependencies

```bash
# Update a specific package
uv pip install --upgrade some-package

# Update all packages (use cautiously)
uv pip install --upgrade -r requirements.txt
```

### Checking Installed Packages

```bash
# List all installed packages
uv pip list

# Show specific package info
uv pip show spindle-kg
```

## Working with Cursor AI Agent

The project has workspace rules configured in `.cursor/rules/uv.mdc` that tell the AI agent to:

- Always use `uv run` for Python commands
- Always use `uv pip install` for package management
- Never use bare `python` or `pip` commands

This ensures consistent environment usage and prevents the "forgot to activate environment" problem common with conda/venv.

## Troubleshooting

### Virtual Environment Not Found

```bash
# Recreate the virtual environment
rm -rf .venv
uv venv
uv pip install -e ".[dev]"
```

### Import Errors

Make sure you installed the project in editable mode:
```bash
uv pip install -e .
```

### Wrong Python Version

Check `.python-version` file and ensure you have that Python version installed:
```bash
cat .python-version  # Should show 3.11
python3.11 --version  # Verify it's installed
```

If needed, install Python 3.11 and recreate the environment:
```bash
# macOS
brew install python@3.11

# Then recreate venv
rm -rf .venv
uv venv
```

### Package Conflicts

UV has excellent dependency resolution, but if you hit conflicts:
```bash
# Clear cache and reinstall
uv cache clean
rm -rf .venv
uv venv
uv pip install -e ".[dev]"
```

## Comparison with Other Tools

### UV vs Pip
- **Speed**: UV is 10-100x faster
- **Reliability**: Better dependency resolution
- **Compatibility**: Drop-in replacement (same commands)

### UV vs Conda
- **Speed**: Much faster package installation
- **Focus**: Python-only (conda handles multiple languages)
- **AI-Friendly**: Simpler commands, easier for AI agents
- **Size**: Smaller, lighter weight

### UV vs Poetry
- **Speed**: UV is faster
- **Simplicity**: Poetry has more features (publishing, etc.)
- **Lock Files**: Poetry has more mature lockfile support
- **Use Case**: UV is great for development, Poetry for publishing

## Advanced Usage

### Multiple Python Versions

```bash
# Create venv with specific Python version
uv venv --python 3.10
uv venv --python 3.11
```

### Sync Environments

```bash
# Install exact versions from requirements
uv pip sync requirements.txt
```

### Compile Requirements

```bash
# Generate pinned requirements (like pip-tools)
uv pip compile pyproject.toml -o requirements.txt
```

## Migration from Conda

If you were previously using the `kgx` conda environment:

1. **Export conda packages** (optional, for reference):
   ```bash
   conda list -n kgx --export > conda_packages.txt
   ```

2. **Remove conda references**:
   - ✅ Already updated `.cursor/rules/` to use uv
   - ✅ Already added `.venv/` to `.gitignore`

3. **Set up uv**:
   ```bash
   uv venv
   uv pip install -e ".[dev]"
   ```

4. **Verify**:
   ```bash
   uv run pytest
   ```

5. **Optional**: Deactivate conda env in your shell profile if set

## Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [UV GitHub](https://github.com/astral-sh/uv)
- [Astral (creators of ruff and uv)](https://astral.sh/)

## Integration with Project

This project uses:
- ✅ `pyproject.toml` - Modern Python project configuration
- ✅ `.python-version` - Specifies Python 3.11
- ✅ `.cursor/rules/uv.mdc` - AI agent instructions
- ✅ `.gitignore` - Excludes `.venv/` and uv artifacts
- ✅ `setup.py` - Maintained for backwards compatibility

All commands in documentation and scripts should use `uv run` prefix.

