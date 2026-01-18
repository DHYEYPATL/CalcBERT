"""
Test script for payment splits API endpoint
Run this to verify the splits endpoint is working correctly
"""
import requests
import json

# Update this URL to match your backend
BACKEND_URL = "http://127.0.0.1:8000"  # Change if using ngrok or different URL

def test_save_split():
    """Test saving a payment split"""
    url = f"{BACKEND_URL}/splits"
    payload = {
        "user_id": "test-user-123",
        "total_amount": 500.0,
        "splits": [
            {"label": "Food", "amount": 200.0},
            {"label": "Transport", "amount": 150.0},
            {"label": "Shopping", "amount": 150.0}
        ]
    }
    
    print("=" * 60)
    print("Testing Save Split Endpoint")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("\n✓ SUCCESS: Split saved successfully!")
            result = response.json()
            if "split_id" in result:
                print(f"  Split ID: {result['split_id']}")
        else:
            print(f"\n✗ FAILED: Status {response.status_code}")
            print(f"  Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n✗ CONNECTION ERROR: Could not connect to backend")
        print(f"  Make sure the backend is running at {BACKEND_URL}")
        print("  Start it with: cd backend && uvicorn app:app --reload")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")

def test_get_latest_split():
    """Test getting the latest split"""
    url = f"{BACKEND_URL}/splits/latest"
    params = {"user_id": "test-user-123"}
    
    print("\n" + "=" * 60)
    print("Testing Get Latest Split Endpoint")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Params: {params}")
    print()
    
    try:
        response = requests.get(url, params=params)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("\n✓ SUCCESS: Retrieved latest split!")
        else:
            print(f"\n✗ FAILED: Status {response.status_code}")
            print(f"  Error: {response.text}")
            
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")

if __name__ == "__main__":
    test_save_split()
    test_get_latest_split()
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    print("\nNow check the database with: python checkDB.py")
