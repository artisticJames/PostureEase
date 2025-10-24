#!/usr/bin/env python3
"""
Simple test script to verify delete functionality works correctly
"""

import requests
import json

def test_delete_endpoint():
    """Test the delete posture record endpoint"""
    
    # Test data
    test_data = {
        'record_id': 1  # Replace with actual record ID from your database
    }
    
    # Test URL (adjust if your server runs on different port)
    url = 'http://localhost:5000/delete-posture-record'
    
    try:
        # Make POST request
        response = requests.post(url, json=test_data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Delete endpoint is working correctly")
        else:
            print("❌ Delete endpoint returned an error")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the Flask app is running.")
    except Exception as e:
        print(f"❌ Error testing delete endpoint: {e}")

def test_clear_all_endpoint():
    """Test the clear all posture history endpoint"""
    
    # Test URL (adjust if your server runs on different port)
    url = 'http://localhost:5000/clear-posture-history'
    
    try:
        # Make POST request
        response = requests.post(url)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Clear all endpoint is working correctly")
        else:
            print("❌ Clear all endpoint returned an error")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the Flask app is running.")
    except Exception as e:
        print(f"❌ Error testing clear all endpoint: {e}")

if __name__ == "__main__":
    print("Testing Posture History Delete Functionality...")
    print("=" * 50)
    test_delete_endpoint()
    print("\n" + "=" * 50)
    print("Testing Clear All History Functionality...")
    print("=" * 50)
    test_clear_all_endpoint()
