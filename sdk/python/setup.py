"""
Setup script for 254Carbon Python SDK.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="carbon254",
    version="1.0.0",
    author="254Carbon",
    author_email="sdk@254carbon.ai",
    description="Official Python SDK for 254Carbon Market Intelligence Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/254carbon/python-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "httpx>=0.25.0",
        "pandas>=2.0.0",
        "pydantic>=2.5.0",
    ],
    extras_require={
        "async": ["aiohttp>=3.9.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.12.0",
            "mypy>=1.7.0",
        ],
    },
)

