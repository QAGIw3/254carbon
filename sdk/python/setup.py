"""
Setup script for the 254Carbon Python SDK.

Notes
-----
- Reads long description and requirements from adjacent files for clarity.
- Declares optional extras for development and examples.
- Packages typed hints via ``py.typed`` (ensure the marker file exists).
"""
from setuptools import setup, find_packages

# Long description for PyPI project page
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Base runtime requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="carbon254",
    version="1.0.0",
    author="254Carbon",
    author_email="sdk@254carbon.ai",
    description="Official Python SDK for 254Carbon Market Intelligence Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/254carbon/python-sdk",
    project_urls={
        "Homepage": "https://github.com/254carbon/python-sdk",
        "Documentation": "https://docs.254carbon.ai/sdk/python",
        "Source": "https://github.com/254carbon/python-sdk",
        "Tracker": "https://github.com/254carbon/python-sdk/issues",
    },
    packages=find_packages(),  # discovers "carbon254" package
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords=[
        "energy", "markets", "trading", "finance", "api", "sdk",
        "real-time", "data", "analysis", "254carbon"
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "mypy>=1.0",
            "pytest-asyncio>=0.21",
            "pytest-cov>=4.0",
        ],
        "examples": [
            "matplotlib>=3.5",
            "seaborn>=0.11",
            "jupyter>=1.0",
        ],
    },
    package_data={
        # Include typing marker for PEP 561 and any markdown metadata
        "carbon254": ["py.typed", "*.md"],
    },
    include_package_data=True,
    zip_safe=False,
)
