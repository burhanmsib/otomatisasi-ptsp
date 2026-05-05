# =========================
# MODULE 2 – FINAL (ROUTE + SEGMENT VISUAL)
# =========================

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw, PolyLineTextPath
from shapely.geometry import LineString
import numpy as np


# =========================
# HELPER – PARSE KOORDINAT
# =========================
def parse_decimal_coordinate(value):
    try:
        parts = str(value).replace(" ", "").split(",")
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None


# =========================
# SPLIT → 4 SEGMENT (UNTUK ANALISIS)
# =========================
def split_route_into_4_segments(points_latlon):

    if len(points_latlon) < 2:
        return None

    line = LineString([(lon, lat) for lat, lon in points_latlon])

    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    result = []

    for f in fractions:
        p = line.interpolate(f, normalized=True)
        result.append((p.y, p.x))

    return result


# =========================
# SPLIT POLYLINE → SEGMENT BENTUK ASLI (UNTUK VISUAL)
# =========================
def split_polyline_into_segments(full_route, n_segments=4):

    if len(full_route) < 2:
        return [full_route]

    line = LineString([(lon, lat) for lat, lon in full_route])
    segments = []

    for i in range(n_segments):

        start_f = i / n_segments
        end_f = (i + 1) / n_segments

        start_d = line.length * start_f
        end_d = line.length * end_f

        coords = []

        for d in np.linspace(start_d, end_d, 25):
            p = line.interpolate(d)
            coords.append((p.y, p.x))

        segments.append(coords)

    return segments


# =========================
# DRAW ROUTE WITH SEGMENTS + ARROW
# =========================
def draw_route_with_segments(map_obj, full_route):

    segments = split_polyline_into_segments(full_route, 4)

    colors = ["red", "blue", "green", "orange"]

    for i, seg in enumerate(segments):

        line = folium.PolyLine(
            locations=seg,
            color=colors[i],
            weight=5,
            opacity=0.9
        ).add_to(map_obj)

        PolyLineTextPath(
            line,
            "➤➤➤",
            repeat=True,
            offset=7,
            attributes={
                "fill": colors[i],
                "font-size": "12"
            }
        ).add_to(map_obj)

    return map_obj


# =========================
# MAIN FUNCTION
# =========================
def process_route_segment_module2_streamlit(row, map_key):

    st.subheader("Mode Input Lokasi")

    st.info(
        "Klik untuk membuat jalur rute (belokan).\n"
        "Klik berkali-kali mengikuti jalur laut.\n"
        "Double klik untuk selesai.\n"
        "Jika salah, gambar ulang."
    )

    mode = st.radio(
        "Pilih Mode",
        ["Gambar Rute", "Titik Tunggal"],
        horizontal=True,
        key=f"mode_{map_key}"
    )

    # =========================
    # MODE TITIK TUNGGAL
    # =========================
    if mode == "Titik Tunggal":

        lat = st.number_input("Latitude", key=f"lat_{map_key}")
        lon = st.number_input("Longitude", key=f"lon_{map_key}")

        if st.button("Simpan Titik", key=f"btn_point_{map_key}"):

            return {
                "tanggal": row.get("Tanggal Koordinat"),
                "awal": (lat, lon),
                "akhir": (lat, lon),
                "titik5": [(lat, lon)],
                "polyline_full": [(lat, lon)]
            }

        return None

    # =========================
    # MODE GAMBAR RUTE
    # =========================
    lat1, lon1 = parse_decimal_coordinate(row.get("Koordinat Awal (Desimal)"))
    lat2, lon2 = parse_decimal_coordinate(row.get("Koordinat Akhir (Desimal)"))

    if None in (lat1, lon1, lat2, lon2):
        st.error("Format koordinat tidak valid")
        return None

    m = folium.Map(
        location=[(lat1 + lat2) / 2, (lon1 + lon2) / 2],
        zoom_start=6,
        tiles="OpenStreetMap"
    )

    folium.Marker([lat1, lon1], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([lat2, lon2], tooltip="End", icon=folium.Icon(color="red")).add_to(m)

    Draw(
        draw_options={
            "polyline": {
                "shapeOptions": {
                    "color": "#1565C0",
                    "weight": 5
                }
            },
            "polygon": False,
            "circle": False,
            "rectangle": False,
            "marker": False,
            "circlemarker": False
        },
        edit_options={"edit": False}
    ).add_to(m)

    output = st_folium(
        m,
        height=800,
        use_container_width=True,
        key=f"draw_map_{map_key}",
        returned_objects=["last_active_drawing"]
    )

    drawing = output.get("last_active_drawing")

    if drawing is None:
        return None

    geom = drawing.get("geometry", {})

    if geom.get("type") != "LineString":
        st.warning("Harus berupa polyline")
        return None

    coords = geom.get("coordinates", [])

    if len(coords) < 2:
        st.error("Minimal 2 titik")
        return None

    # 🔥 polyline asli (TIDAK DIUBAH)
    full_route = [(pt[1], pt[0]) for pt in coords]

    # 🔥 titik untuk analisis
    titik5 = split_route_into_4_segments(full_route)

    # =========================
    # MAP FINAL (SEGMENTED + ARROW)
    # =========================
    m2 = folium.Map(
        location=[(lat1 + lat2) / 2, (lon1 + lon2) / 2],
        zoom_start=6,
        tiles="OpenStreetMap"
    )

    m2 = draw_route_with_segments(m2, full_route)

    st.success("Rute tersimpan (mengikuti jalur & segmented)")

    st_folium(
        m2,
        height=800,
        use_container_width=True,
        key=f"final_map_{map_key}"
    )

    return {
        "tanggal": row.get("Tanggal Koordinat"),
        "awal": (lat1, lon1),
        "akhir": (lat2, lon2),
        "titik5": titik5,
        "polyline_full": full_route
    }
