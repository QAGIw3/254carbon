#!/bin/bash

# 254Carbon Excel Add-in Build Script
# Builds and packages the Excel add-in for local development

set -e

echo "🔨 Building 254Carbon Excel Add-in..."

# Check if .NET SDK is installed
if ! command -v dotnet > /dev/null 2>&1; then
    echo "❌ .NET SDK is not installed. Please install .NET 6.0 SDK or later."
    exit 1
fi

# Navigate to excel-addin directory
cd "$(dirname "$0")"

echo "📦 Restoring packages..."
dotnet restore

echo "🔨 Building add-in..."
dotnet build --configuration Release

echo "📋 Build Summary:"
echo "   • Project: 254Carbon Excel Add-in"
echo "   • Target: .NET 6.0 Windows"
echo "   • Output: bin/Release/net6.0-windows/"
echo ""

echo "📁 Generated Files:"
echo "   • 254Carbon.dll (main assembly)"
echo "   • 254Carbon-AddIn.xll (Excel add-in)"
echo "   • 254Carbon-AddIn.dna (configuration)"
echo ""

echo "🎯 Installation Steps:"
echo "   1. Open Excel"
echo "   2. Go to File → Options → Add-ins"
echo "   3. Click 'Go' next to 'Manage: Excel Add-ins'"
echo "   4. Click 'Browse' and select: bin/Release/net6.0-windows/254Carbon-AddIn.xll"
echo "   5. Check the box next to '254Carbon' and click OK"
echo ""

echo "🔧 Development Setup:"
echo "   export CARBON254_LOCAL_DEV=true"
echo "   export CARBON254_API_URL=http://localhost:8000"
echo "   export CARBON254_API_KEY=dev-key"
echo ""

echo "✅ Build complete! Ready for testing."
