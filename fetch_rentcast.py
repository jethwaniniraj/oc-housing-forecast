import requests
import json
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("RENTCAST_API_KEY")

RENTCAST_BASE_URL = "https://api.rentcast.io/v1/listings/sale"

def fetch_listings_by_zip(zip_code, limit=12):
    """
    Fetches active property listings for sale from Rentcast API.
    
    Args:
        zip_code: The 5-digit US ZIP code to search
        limit: Maximum number of listings to return (default 12)
    
    Returns:
        List of property listings or empty list if error
    """
    
    if not API_KEY:
        print("❌ Error: API key not found. Check your .env file.")
        return []
    
    headers = {
        "accept": "application/json",
        "X-Api-Key": API_KEY
    }
    
    params = {
        "zipCode": zip_code,
        "status": "Active",
        "limit": limit
    }
    
    try:
        print(f"🔍 Fetching listings for ZIP: {zip_code}...")
        response = requests.get(RENTCAST_BASE_URL, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print(f"⚠️ No listings found for ZIP: {zip_code}")
            return []
        
        # Format the data for our template
        listings = []
        for property in data:
            listing = {
                "price": property.get("price", 0),
                "addressLine1": property.get("formattedAddress", "Address unavailable"),
                "bedrooms": property.get("bedrooms", "N/A"),
                "bathrooms": property.get("bathrooms", "N/A"),
                "squareFootage": property.get("squareFootage", "N/A"),
                "propertyType": property.get("propertyType", "Unknown"),
                "listingType": property.get("listingType", "Sale"),
                "daysOnMarket": property.get("daysOnMarket", "N/A"),
                "latitude": property.get("latitude"),
                "longitude": property.get("longitude"),
                "photoUrl": property.get("photoUrl", None)
            }
            listings.append(listing)
        
        print(f"✅ Found {len(listings)} listings!")
        return listings
    
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if response.status_code == 401:
            print("   → Invalid API key. Please check your Rentcast API key.")
        elif response.status_code == 429:
            print("   → Rate limit exceeded. Please wait before making more requests.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return []


def save_listings_to_json(zip_code, listings, filename="data.json"):
    """
    Saves fetched listings to a JSON file for caching.
    """
    try:
        # Load existing data or create new
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = {}
        
        # Update with new listings
        data[zip_code] = listings
        
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved {len(listings)} listings to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving to JSON: {e}")
        return False


# Test the API if running directly
if __name__ == "__main__":
    test_zip = input("Enter a ZIP code to test (e.g., 92618): ").strip()
    listings = fetch_listings_by_zip(test_zip)
    
    if listings:
        print(f"\n📋 Sample listing:")
        print(f"   Address: {listings[0]['addressLine1']}")
        print(f"   Price: ${listings[0]['price']:,}")
        print(f"   Beds/Baths: {listings[0]['bedrooms']} bd | {listings[0]['bathrooms']} ba")
        
        # Save to JSON
        save_listings_to_json(test_zip, listings)