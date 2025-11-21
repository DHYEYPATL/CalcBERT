"""
API Performance Benchmarking Script
Measures latency and throughput of the CalcBERT backend.
"""

import time
import requests
import statistics
from typing import List, Dict, Any


# Configuration
BASE_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{BASE_URL}/predict"

# Test data
TEST_TRANSACTIONS = [
    "STARBCKS #103",
    "AMAZON.COM PURCHASE",
    "UBER TRIP 12:30PM",
    "WALMART SUPERCENTER",
    "NETFLIX SUBSCRIPTION",
    "SHELL GAS STATION",
    "MCDONALDS #4521",
    "APPLE.COM/BILL",
    "CVS PHARMACY",
    "TARGET STORE"
]


def benchmark_predict(n_requests: int = 100) -> Dict[str, Any]:
    """
    Benchmark the /predict endpoint.
    
    Args:
        n_requests: Number of requests to make
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking /predict endpoint with {n_requests} requests")
    print(f"{'='*60}\n")
    
    latencies = []
    errors = 0
    
    start_time = time.time()
    
    for i in range(n_requests):
        # Cycle through test transactions
        text = TEST_TRANSACTIONS[i % len(TEST_TRANSACTIONS)]
        
        req_start = time.time()
        try:
            response = requests.post(
                PREDICT_URL,
                json={"text": text},
                timeout=10
            )
            req_end = time.time()
            
            if response.status_code == 200:
                latencies.append((req_end - req_start) * 1000)  # Convert to ms
            else:
                errors += 1
                print(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            errors += 1
            print(f"Request failed: {e}")
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{n_requests} requests completed")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    if latencies:
        results = {
            "total_requests": n_requests,
            "successful_requests": len(latencies),
            "failed_requests": errors,
            "total_time_seconds": round(total_time, 2),
            "requests_per_second": round(n_requests / total_time, 2),
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 2),
                "median": round(statistics.median(latencies), 2),
                "min": round(min(latencies), 2),
                "max": round(max(latencies), 2),
                "p95": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) > 20 else None,
                "p99": round(statistics.quantiles(latencies, n=100)[98], 2) if len(latencies) > 100 else None,
            }
        }
    else:
        results = {
            "total_requests": n_requests,
            "successful_requests": 0,
            "failed_requests": errors,
            "error": "All requests failed"
        }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print benchmark results in a readable format."""
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Total Requests:       {results['total_requests']}")
    print(f"Successful:           {results['successful_requests']}")
    print(f"Failed:               {results['failed_requests']}")
    
    if 'total_time_seconds' in results:
        print(f"\nTotal Time:           {results['total_time_seconds']} seconds")
        print(f"Throughput:           {results['requests_per_second']} req/sec")
        
        print(f"\nLatency (milliseconds):")
        print(f"  Mean:               {results['latency_ms']['mean']} ms")
        print(f"  Median:             {results['latency_ms']['median']} ms")
        print(f"  Min:                {results['latency_ms']['min']} ms")
        print(f"  Max:                {results['latency_ms']['max']} ms")
        
        if results['latency_ms']['p95']:
            print(f"  P95:                {results['latency_ms']['p95']} ms")
        if results['latency_ms']['p99']:
            print(f"  P99:                {results['latency_ms']['p99']} ms")
    
    print(f"\n{'='*60}\n")


def test_endpoint_availability() -> bool:
    """Test if the backend is available."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ Backend is available at {BASE_URL}")
            return True
        else:
            print(f"✗ Backend returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Backend not available: {e}")
        print(f"  Make sure the server is running at {BASE_URL}")
        return False


def main():
    """Main benchmark function."""
    print("\nCalcBERT Backend Performance Benchmark")
    
    # Check if backend is available
    if not test_endpoint_availability():
        print("\nPlease start the backend server first:")
        print("  cd backend")
        print("  uvicorn app:app --reload --host 127.0.0.1 --port 8000")
        return
    
    # Run benchmark
    results = benchmark_predict(n_requests=50)
    
    # Print results
    print_results(results)
    
    # Save results to file
    import json
    output_file = "metrics/bench_results.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Could not save results: {e}")


if __name__ == "__main__":
    main()
