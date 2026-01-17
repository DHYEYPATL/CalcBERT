"""
Smoke Test Suite for CalcBERT v2 Backend
Tests all critical endpoints and components.
"""

import requests
import json
import sys
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name):
    print(f"\n{Colors.BLUE}🧪 Testing: {name}{Colors.END}")

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def test_health():
    """Test health endpoint."""
    print_test("Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                print_success(f"Health check passed: {data}")
                return True
        print_error(f"Health check failed: {response.status_code}")
        return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False

def test_root():
    """Test root endpoint."""
    print_test("Root Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            endpoints = data.get("endpoints", {})
            required = ["prepay_check", "summary_today", "subscriptions", "alerts"]
            missing = [e for e in required if e not in endpoints]
            if not missing:
                print_success(f"All v2 endpoints registered: {required}")
                return True
            else:
                print_error(f"Missing endpoints: {missing}")
                return False
        print_error(f"Root endpoint failed: {response.status_code}")
        return False
    except Exception as e:
        print_error(f"Root endpoint error: {e}")
        return False

def test_prepay_check():
    """Test prepay check endpoint."""
    print_test("Prepay Check API")
    
    test_cases = [
        {
            "name": "Normal Business Expense",
            "payload": {
                "merchant": "Swiggy",
                "amount": 1200.0,
                "note": "client meeting + dinner",
                "upi_id": "swiggy@upi",
                "user_role": "employee",
                "user_id": "smoke-test-1",
                "date": "2025-12-24"
            },
            "expected_decision": ["allow", "warn"]
        },
        {
            "name": "High Amount Warning",
            "payload": {
                "merchant": "Hotel Taj",
                "amount": 55000.0,
                "note": "conference accommodation",
                "upi_id": "taj@upi",
                "user_role": "employee",
                "user_id": "smoke-test-2",
                "date": "2025-12-24"
            },
            "expected_decision": ["warn", "block"]
        },
        {
            "name": "Employee Alcohol Block",
            "payload": {
                "merchant": "Wine Shop",
                "amount": 2500.0,
                "note": "team celebration",
                "upi_id": "wine@upi",
                "user_role": "employee",
                "user_id": "smoke-test-3",
                "date": "2025-12-24"
            },
            "expected_decision": ["block", "warn"]
        }
    ]
    
    passed = 0
    for test in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/prepay/check",
                json=test["payload"],
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                decision = data.get("decision")
                
                if decision in test["expected_decision"]:
                    print_success(f"{test['name']}: {decision} ✓")
                    passed += 1
                else:
                    print_warning(f"{test['name']}: Got {decision}, expected {test['expected_decision']}")
                    passed += 0.5  # Partial credit
            else:
                print_error(f"{test['name']}: HTTP {response.status_code}")
        except Exception as e:
            print_error(f"{test['name']}: {e}")
    
    return passed == len(test_cases)

def test_prepay_history():
    """Test prepay history endpoint."""
    print_test("Prepay History API")
    try:
        response = requests.get(
            f"{BASE_URL}/prepay/history?user_id=smoke-test-1&limit=10",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok" and "history" in data:
                print_success(f"History retrieved: {data['count']} records")
                return True
        print_error(f"History failed: {response.status_code}")
        return False
    except Exception as e:
        print_error(f"History error: {e}")
        return False

def test_prepay_stats():
    """Test prepay stats endpoint."""
    print_test("Prepay Stats API")
    try:
        response = requests.get(f"{BASE_URL}/prepay/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok" and "total_checks" in data:
                print_success(f"Stats retrieved: {data['total_checks']} total checks")
                return True
        print_error(f"Stats failed: {response.status_code}")
        return False
    except Exception as e:
        print_error(f"Stats error: {e}")
        return False

def test_summary_today():
    """Test daily summary endpoint."""
    print_test("Daily Summary API")
    try:
        response = requests.get(f"{BASE_URL}/summary/today", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok" and "summary" in data:
                print_success(f"Summary retrieved: {data['summary'].get('total_transactions', 0)} transactions today")
                return True
        print_error(f"Summary failed: {response.status_code}")
        return False
    except Exception as e:
        print_error(f"Summary error: {e}")
        return False

def test_subscriptions():
    """Test subscriptions endpoint."""
    print_test("Subscriptions API")
    try:
        response = requests.get(f"{BASE_URL}/summary/subscriptions", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok" and "subscriptions" in data:
                print_success(f"Subscriptions retrieved: {data['count']} subscriptions")
                return True
        print_error(f"Subscriptions failed: {response.status_code}")
        return False
    except Exception as e:
        print_error(f"Subscriptions error: {e}")
        return False

def test_alerts():
    """Test alerts endpoint."""
    print_test("Alerts API")
    try:
        response = requests.get(f"{BASE_URL}/summary/alerts", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok" and "alerts" in data:
                print_success(f"Alerts retrieved: {data['count']} alerts")
                return True
        print_error(f"Alerts failed: {response.status_code}")
        return False
    except Exception as e:
        print_error(f"Alerts error: {e}")
        return False

def run_all_tests():
    """Run all smoke tests."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}CalcBERT v2 Backend - Smoke Test Suite{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Prepay Check", test_prepay_check),
        ("Prepay History", test_prepay_history),
        ("Prepay Stats", test_prepay_stats),
        ("Daily Summary", test_summary_today),
        ("Subscriptions", test_subscriptions),
        ("Alerts", test_alerts),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Test Summary{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"{name:.<40} {status}")
    
    print(f"\n{Colors.BLUE}Total: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}🎉 All tests passed! System is ready.{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.RED}⚠️  Some tests failed. Please check the logs.{Colors.END}\n")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
