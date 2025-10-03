#!/bin/bash
# Script to build and publish the 254Carbon Python SDK to PyPI

set -e

echo "🚀 Building 254Carbon Python SDK for PyPI..."

# Clean previous builds
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/

# Build the package
python setup.py sdist bdist_wheel

# Check the built files
echo "📦 Built files:"
ls -la dist/

# Optional: Test the package installation locally
echo "🧪 Testing local installation..."
pip install dist/carbon254-1.0.0.tar.gz

# Publish to PyPI (requires credentials)
echo "📤 Publishing to PyPI..."
echo "⚠️  Make sure you have PyPI credentials configured:"
echo "   - Set TWINE_USERNAME and TWINE_PASSWORD environment variables"
echo "   - Or use 'twine upload dist/*' manually"

# Uncomment the following line when ready to publish
# twine upload dist/*

echo "✅ Build complete! Ready for PyPI publishing."
echo "   To publish: twine upload dist/*"
