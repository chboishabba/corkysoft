import googlemaps
from datetime import datetime

gmaps = googlemaps.Client(key="YOUR_API_KEY")

def driving_distance_km_google(origin_city, dest_city):
    # You can pass city names directly; Google will geocode them
    res = gmaps.distance_matrix(
        origins=[origin_city],
        destinations=[dest_city],
        mode="driving",
        departure_time=datetime.now()  # enables traffic-aware duration_in_traffic
    )
    el = res["rows"][0]["elements"][0]
    meters = el["distance"]["value"]
    seconds = el.get("duration_in_traffic", el["duration"])["value"]
    return meters / 1000.0, seconds / 3600.0  # (km, hours)

km, hours = driving_distance_km_google("Melbourne, AU", "Sydney, AU")
print(km, hours)
