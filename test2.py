import openrouteservice as ors

# Replace with your free ORS API key
client = ors.Client(key="eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImY1Mjg5NjdmYzIxZTQ1MjNhY2JiMDMxZWQyMzY5NmM2IiwiaCI6Im11cm11cjY0In0=")

hourlyRate = 200

def driving_distance_km_ors(origin, destination, country="Australia"):
    # Geocode city names
    o = client.pelias_search(text=f"{origin}, {country}")["features"][0]["geometry"]["coordinates"]
    d = client.pelias_search(text=f"{destination}, {country}")["features"][0]["geometry"]["coordinates"]

    # Request driving route
    route = client.directions(
        coordinates=[o, d],
        profile="driving-car",
        format="json"
    )

    summary = route["routes"][0]["summary"]
    meters = summary["distance"]
    seconds = summary["duration"]

    return meters / 1000.0, seconds / 3600.0  # (km, hours)
billEstimate = hourlyRate * hours
# Example usage
km, hours = driving_distance_km_ors("Melbourne", "Sydney")
print(f"Distance: {km:.1f} km, Duration: {hours:.1f} hours Billable: {billEstimate:.1f}")
