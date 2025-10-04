#!/bin/bash
# Generate sample analytics reports using platform services

# 254Carbon Platform - Report Generation Script
# Tests and demonstrates report generation functionality

set -e

echo "📊 Testing report generation..."

# Check if report service is running
if ! curl -s "http://localhost:8004/health" > /dev/null 2>&1; then
    echo "❌ Report service is not running. Please start it first."
    echo "💡 Run: docker-compose up -d report-service"
    exit 1
fi

echo "✅ Report service is available"

# Test report generation for different markets
markets=("power" "gas")
report_types=("monthly_brief" "custom")

echo ""
echo "🧪 Generating test reports..."

for market in "${markets[@]}"; do
    for report_type in "${report_types[@]}"; do
        echo ""
        echo "Generating $report_type report for $market market..."

        # Generate report
        response=$(curl -s -X POST "http://localhost:8004/api/v1/reports" \
            -H "Content-Type: application/json" \
            -d '{
                "report_type": "'$report_type'",
                "market": "'$market'",
                "as_of_date": "'$(date +%Y-%m-%d)'"',
                "format": "pdf"
            }')

        # Extract report ID
        report_id=$(echo "$response" | grep -o '"report_id":"[^"]*"' | cut -d'"' -f4)

        if [ -n "$report_id" ]; then
            echo "✅ Report generation started: $report_id"

            # Wait for completion
            echo "⏳ Waiting for report completion..."
            timeout=60
            while [ $timeout -gt 0 ]; do
                status_response=$(curl -s "http://localhost:8004/api/v1/reports/$report_id")
                status=$(echo "$status_response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

                if [ "$status" = "completed" ]; then
                    download_url=$(echo "$status_response" | grep -o '"download_url":"[^"]*"' | cut -d'"' -f4)
                    echo "✅ Report completed: $download_url"
                    break
                elif [ "$status" = "failed" ]; then
                    echo "❌ Report generation failed"
                    break
                fi

                sleep 2
                timeout=$((timeout - 2))
            done

            if [ $timeout -le 0 ] && [ "$status" != "completed" ]; then
                echo "⏰ Report generation timed out"
            fi

        else
            echo "❌ Failed to start report generation"
            echo "Response: $response"
        fi
    done
done

echo ""
echo "📋 Report Generation Summary:"
echo ""
echo "🎯 Generated reports for:"
for market in "${markets[@]}"; do
    for report_type in "${report_types[@]}"; do
        echo "   • $report_type report - $market market"
    done
done

echo ""
echo "💡 Tips:"
echo "   • Check report status: curl http://localhost:8004/api/v1/reports/{report_id}"
echo "   • Download reports from the provided URLs"
echo "   • View sample reports in MinIO at http://localhost:9001/reports/"
echo "   • Test different parameters and date ranges"
echo ""
echo "🚀 Report generation testing complete!"
