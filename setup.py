"""Setup configuration for Spindle package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="spindle-kg",
    version="0.1.0",
    author="Spindle Contributors",
    description="A tool for real-time extraction of knowledge graphs from multimodal data using BAML and LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spindle",
    packages=find_packages(exclude=["tests", "tests.*", "demos", "demos.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "freezegun>=1.4.0",
            "hypothesis>=6.98.0",
        ],
        "graph": [
            "kuzu>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add CLI commands here if needed in the future
        ],
    },
    include_package_data=True,
    package_data={
        "spindle": ["py.typed"],
    },
)

