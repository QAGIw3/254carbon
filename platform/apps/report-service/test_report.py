#!/usr/bin/env python3
"""
Test script for report generation functionality.
"""

import asyncio
import requests
from datetime import date, datetime

async def test_report_generation():
    """Test the report generation endpoints."""

    base_url = "http://localhost:8004"

    # Test health endpoint
    try:
        health_response = requests.get(f"{base_url}/health")
        if health_response.status_code == 200:
            print("✅ Report service is healthy")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return

    # Test report generation
    print("\n📊 Testing report generation...")

    test_cases = [
        {
            "report_type": "monthly_brief",
            "market": "power",
            "as_of_date": date.today().isoformat(),
            "format": "pdf"
        },
        {
            "report_type": "custom",
            "market": "gas",
            "as_of_date": date.today().isoformat(),
            "format": "html"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['report_type']} report for {test_case['market']}")

        try:
            # Generate report
            response = requests.post(
                f"{base_url}/api/v1/reports",
                json=test_case,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                report_id = result.get("report_id")
                status = result.get("status")

                print(f"✅ Report generation started: {report_id} (status: {status})")

                if status == "completed" and result.get("download_url"):
                    download_url = result["download_url"]
                    print(f"📄 Report available at: {download_url}")

                    # Test download
                    try:
                        download_response = requests.get(download_url, timeout=10)
                        if download_response.status_code == 200:
                            print(f"✅ Report download successful ({len(download_response.content)} bytes)")
                        else:
                            print(f"❌ Report download failed: {download_response.status_code}")
                    except Exception as e:
                        print(f"❌ Report download error: {e}")

                else:
                    print(f"⚠️  Report not immediately available (status: {status})")

            else:
                print(f"❌ Report generation failed: {response.status_code}")
                print(f"Error: {response.text}")

        except Exception as e:
            print(f"❌ Report generation error: {e}")

    # Test report status check
    print("\n🔍 Testing report status check...")

    # Get a report ID from previous generation (mock for demo)
    test_report_id = "test-report-123"

    try:
        status_response = requests.get(f"{base_url}/api/v1/reports/{test_report_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"✅ Report status check: {status_data.get('status', 'unknown')}")
        else:
            print(f"❌ Report status check failed: {status_response.status_code}")
    except Exception as e:
        print(f"❌ Report status check error: {e}")

    print("\n🎉 Report generation testing completed!")

if __name__ == "__main__":
    print("🚀 Starting report generation tests...")
    asyncio.run(test_report_generation())
