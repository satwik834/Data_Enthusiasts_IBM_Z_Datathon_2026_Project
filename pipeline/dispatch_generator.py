"""
Ambulance Fleet Dispatch & Geospatial Route Simulator
Simulates real-time ambulance transit, GPS telemetry, ETA, and hospital trauma center routing.
"""

import math
import time
from typing import Dict, Any, List

# Major Hospital Trauma Center Reference Locations (e.g. Metro Area)
HOSPITAL_TRAUMA_CENTERS = [
    {"name": "Central General Hospital & Trauma Center", "lat": 40.7589, "lon": -73.9851, "icu_beds_free": 4},
    {"name": "St. Jude Metropolitan Medical Center", "lat": 40.7484, "lon": -73.9857, "icu_beds_free": 2},
    {"name": "Memorial University Health Emergency Wing", "lat": 40.7614, "lon": -73.9776, "icu_beds_free": 6}
]


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


class AmbulanceDispatchUnit:
    def __init__(
        self,
        unit_id: str,
        start_lat: float = 40.7128,
        start_lon: float = -74.0060,
        dest_hospital: Dict[str, Any] = None
    ):
        self.unit_id = unit_id
        self.current_lat = start_lat
        self.current_lon = start_lon
        self.dest_hospital = dest_hospital or HOSPITAL_TRAUMA_CENTERS[0]
        self.speed_kmh = 55.0
        self.siren_active = False
        self.total_distance_km = haversine_distance_km(
            start_lat, start_lon, self.dest_hospital["lat"], self.dest_hospital["lon"]
        )
        self.progress = 0.0

    def step(self, dt_seconds: float = 2.0, high_priority: bool = False) -> Dict[str, Any]:
        """Steps the ambulance vehicle along its transit trajectory towards the hospital."""
        self.siren_active = high_priority
        effective_speed = 75.0 if self.siren_active else self.speed_kmh

        # Distance moved in this step
        distance_moved_km = (effective_speed / 3600.0) * dt_seconds
        if self.total_distance_km > 0.01:
            self.progress = min(1.0, self.progress + (distance_moved_km / self.total_distance_km))

        # Linear interpolation between origin and destination
        target_lat = self.dest_hospital["lat"]
        target_lon = self.dest_hospital["lon"]
        self.current_lat = self.current_lat + (target_lat - self.current_lat) * min(1.0, distance_moved_km / max(0.01, self.total_distance_km))
        self.current_lon = self.current_lon + (target_lon - self.current_lon) * min(1.0, distance_moved_km / max(0.01, self.total_distance_km))

        remaining_km = haversine_distance_km(self.current_lat, self.current_lon, target_lat, target_lon)
        eta_minutes = (remaining_km / max(10.0, effective_speed)) * 60.0

        return {
            "unit_id": self.unit_id,
            "timestamp": time.time(),
            "lat": round(self.current_lat, 5),
            "lon": round(self.current_lon, 5),
            "speed_kmh": round(effective_speed, 1),
            "siren_active": self.siren_active,
            "destination_hospital": self.dest_hospital["name"],
            "remaining_distance_km": round(remaining_km, 2),
            "eta_minutes": round(eta_minutes, 1),
            "icu_beds_free": self.dest_hospital["icu_beds_free"],
            "transit_progress_pct": round(self.progress * 100.0, 1)
        }
