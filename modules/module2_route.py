# =========================
# MODULE 2 – FINAL CLEAN
# =========================

import streamlit as st
from streamlit_folium import st_folium
import folium
import json
from folium.plugins import Draw
from shapely.geometry import LineString
from io import BytesIO
from timezonefinder import TimezoneFinder

tf = TimezoneFinder()

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
# Konversi Zona Waktu
# =========================
def get_timezone(lat, lon):
    """
    Menentukan zona waktu Indonesia berdasarkan koordinat.
    Mengembalikan:
    WIB / WITA / WIT
    """

    tz = tf.timezone_at(lat=lat, lng=lon)

    mapping = {
        "Asia/Jakarta": "WIB",
        "Asia/Pontianak": "WIB",

        "Asia/Makassar": "WITA",

        "Asia/Jayapura": "WIT"
    }

    return mapping.get(tz, "WIB")

# =========================
# SPLIT → 4 SEGMENT (5 TITIK)
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
# Download Json
# =========================

def create_geojson(full_route, titik5):

    features = []

    # Garis rute
    features.append({
        "type": "Feature",
        "properties": {
            "name": "Route"
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [lon, lat] for lat, lon in full_route
            ]
        }
    })

    # Titik sampling
    for i, (lat, lon) in enumerate(titik5, start=1):

        features.append({
            "type": "Feature",
            "properties": {
                "point": i
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        })

    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": features
        },
        indent=2
    )


# =========================
# MAIN FUNCTION
# =========================
def process_route_segment_module2_streamlit(row, map_key):

    st.subheader("Mode Input Lokasi")

    # st.info(
    #     "Klik beberapa titik untuk membuat jalur rute (belokan).\n"
    #     "Semakin banyak titik, semakin detail jalurnya.\n"
    #     "Double klik untuk menyelesaikan."
    # )

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
        tz = get_timezone(lat, lon)

        if st.button("Simpan Titik", key=f"btn_point_{map_key}"):

            return {
                "tanggal": row.get("Tanggal Koordinat"),
                "awal": (lat, lon),
                "akhir": (lat, lon),
                "titik5": [(lat, lon)],
                "polyline_full": [(lat, lon)],
                "tz": tz
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

    # =========================
    # LAYOUT
    # =========================
    col_draw, col_preview = st.columns([1, 1])

    # =========================
    # MAP DRAW
    # =========================
    m = folium.Map(
        location=[(lat1 + lat2) / 2, (lon1 + lon2) / 2],
        zoom_start=6,
        tiles="OpenStreetMap"
    )

    # 🔥 START
    folium.Marker(
        [lat1, lon1],
        tooltip="Start Point",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    # 🔥 END
    folium.Marker(
        [lat2, lon2],
        tooltip="End Point",
        icon=folium.Icon(color="red", icon="flag")
    ).add_to(m)

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
        }
    ).add_to(m)

    with col_draw:

        st.markdown("### 🗺️ Gambar Rute")
    
        output = st_folium(
            m,
            height=800,
            use_container_width=True,
            key=f"draw_map_{map_key}",
            returned_objects=["last_active_drawing"]
        )

    drawing = output.get("last_active_drawing")

    with col_preview:
    
        st.markdown("### 📍 Preview Rute")
    
        if drawing is None:
            st.info("Belum ada rute yang digambar.")
    
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

    # 🔥 POLYLINE ASLI
    full_route = [(pt[1], pt[0]) for pt in coords]

    # 🔥 5 TITIK (SEGMENT)
    titik5 = split_route_into_4_segments(full_route)

    rep_lat, rep_lon = titik5[2]

    tz = get_timezone(rep_lat, rep_lon)

    # =========================
    # MAP FINAL (CLEAN)
    # =========================
    m2 = folium.Map(
        location=[(lat1 + lat2) / 2, (lon1 + lon2) / 2],
        zoom_start=6,
        tiles="OpenStreetMap"
    )

    # 🔥 GARIS RUTE
    folium.PolyLine(
        locations=full_route,
        color="#1565C0",
        weight=6
    ).add_to(m2)

    # 🔥 START
    folium.Marker(
        [lat1, lon1],
        tooltip="Start Point",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m2)

    # 🔥 END
    folium.Marker(
        [lat2, lon2],
        tooltip="End Point",
        icon=folium.Icon(color="red", icon="flag")
    ).add_to(m2)

    # 🔥 TITIK 1–5
    for i, (lat, lon) in enumerate(titik5, start=1):
        folium.Marker(
            [lat, lon],
            tooltip=f"Titik {i}",
            icon=folium.DivIcon(html=f"""
                <div style="
                    background:#0D47A1;
                    color:white;
                    border-radius:50%;
                    width:28px;
                    height:28px;
                    text-align:center;
                    line-height:28px;
                    font-weight:bold;
                    border:2px solid white;
                ">
                    {i}
                </div>
            """)
        ).add_to(m2)

    with col_preview:
    
        st_folium(
            m2,
            height=800,
            use_container_width=True,
            key=f"final_map_{map_key}"
        )
        
    st.success("Rute tersimpan")

    geojson = create_geojson(full_route, titik5)

    st.download_button(
        "📍 Download Route (.geojson)",
        data=geojson,
        file_name=f"route_{map_key}.geojson",
        mime="application/geo+json",
        key=f"download_geojson_{map_key}"
    )

    # =========================
    # DOWNLOAD HTML
    # =========================
    html = m2.get_root().render()
    
    st.download_button(
        label="🌐 Download Route Map (.html)",
        data=html,
        file_name=f"Route_{map_key}.html",
        mime="text/html",
        key=f"download_html_{map_key}"
    )

    return {
        "tanggal": row.get("Tanggal Koordinat"),
        "awal": (lat1, lon1),
        "akhir": (lat2, lon2),
        "titik5": titik5,
        "polyline_full": full_route,
        "tz": tz,
    
        # Simpan file rute agar tetap tersedia setelah rerun
        "geojson": geojson,
        "map_html": html
    }
