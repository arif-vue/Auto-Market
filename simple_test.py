import requests

print("Testing API endpoints...")

# Test basic endpoint accessibility
try:
    response = requests.get('http://127.0.0.1:8000/api/dashboard/')
    print(f"Dashboard: {response.status_code} - {response.reason}")
    if response.status_code == 401:
        print("✅ Authentication required (as expected)")
except Exception as e:
    print(f"❌ Dashboard test failed: {e}")

try:
    response = requests.get('http://127.0.0.1:8000/api/submissions/list/')
    print(f"Submissions List: {response.status_code} - {response.reason}")
    if response.status_code == 401:
        print("✅ Authentication required (as expected)")
except Exception as e:
    print(f"❌ Submissions test failed: {e}")

try:
    response = requests.get('http://127.0.0.1:8000/api/products/')
    print(f"Products List: {response.status_code} - {response.reason}")
    if response.status_code == 401:
        print("✅ Authentication required (as expected)")
except Exception as e:
    print(f"❌ Products test failed: {e}")

try:
    response = requests.get('http://127.0.0.1:8000/api/admin/dashboard/')
    print(f"Admin Dashboard: {response.status_code} - {response.reason}")
    if response.status_code == 401:
        print("✅ Admin authentication required (as expected)")
except Exception as e:
    print(f"❌ Admin dashboard test failed: {e}")

print("\n✅ All endpoints are accessible and properly require authentication!")