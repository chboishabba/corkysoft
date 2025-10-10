import sqlite3, folium

DB_PATH = "routes.db"

conn = sqlite3.connect(DB_PATH)
rows = conn.execute("""
    SELECT id, origin, destination,
           origin_lon, origin_lat, dest_lon, dest_lat,
           COALESCE(origin_resolved, origin),
           COALESCE(destination_resolved, destination)
    FROM jobs
    WHERE origin_lon IS NOT NULL AND dest_lon IS NOT NULL
""").fetchall()
conn.close()

m = folium.Map(location=[-25.0, 135.0], zoom_start=4)  # center on Australia

for jid, o, d, olon, olat, dlon, dlat, or_res, dr_res in rows:
    # markers
    folium.Marker([olat, olon], popup=f"{o} → {d}\n{or_res}", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker([dlat, dlon], popup=f"{o} → {d}\n{dr_res}", icon=folium.Icon(color="red")).add_to(m)
    # route line (straight line between coords, not full road path)
    folium.PolyLine([[olat, olon], [dlat, dlon]], color="green", weight=2.5, opacity=0.8).add_to(m)

m.save("routes_map.html")
print("Map saved to routes_map.html")
