# =========================
# MODULE 2 – ROUTE ENGINE (FINAL FIXED)
# =========================

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
from shapely.geometry import LineString

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
# 🔥 SPLIT MENJADI 4 SEGMENT (5 TITIK)
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
# MARKER STYLE
# =========================
def numbered_marker(lat, lon, number):
    html = f"""
    <div style="
        background-color:#0D47A1;
        color:white;
        border-radius:50%;
        width:30px;
        height:30px;
        text-align:center;
        font-weight:bold;
        font-size:13px;
        line-height:30px;
        border:2px solid white;
        box-shadow:0 0 5px rgba(0,0,0,0.6);
    ">
        {number}
    </div>
    """
    return folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=html),
        tooltip=f"Titik {number}"
    )


# =========================
# MAIN FUNCTION
# =========================
def process_route_segment_module2_streamlit(row, map_key):

    st.subheader("Mode Input Lokasi")

    # =========================
    # MODE PILIHAN
    # =========================
    mode = st.radio(
        "Pilih Mode",
        ["Gambar Rute", "Titik Tunggal"],
        horizontal=True,
        key=f"mode_{map_key}"
    )

    # =========================
    # MODE 1: TITIK TUNGGAL
    # =========================
    if mode == "Titik Tunggal":

        st.info("Gunakan ini jika hanya 1 koordinat")

        lat = st.number_input("Latitude", key=f"lat_{map_key}")
        lon = st.number_input("Longitude", key=f"lon_{map_key}")

        if st.button("Simpan Titik", key=f"btn_point_{map_key}"):

            st.success("✅ Titik berhasil disimpan")

            return {
                "tanggal": row.get("Tanggal Koordinat"),
                "awal": (lat, lon),
                "akhir": (lat, lon),
                "titik5": [(lat, lon)],
                "polyline_full": [(lat, lon)]
            }

        return None

    # =========================
    # MODE 2: GAMBAR RUTE
    # =========================

    lat1, lon1 = parse_decimal_coordinate(row.get("Koordinat Awal (Desimal)"))
    lat2, lon2 = parse_decimal_coordinate(row.get("Koordinat Akhir (Desimal)"))

    if None in (lat1, lon1, lat2, lon2):
        st.error("Format koordinat desimal tidak valid.")
        return None

    st.caption(f"{row.get('Koordinat Awal')} ➜ {row.get('Koordinat Akhir')}")

    # =========================
    # MAP DRAW
    # =========================
    m = folium.Map(
        location=[(lat1 + lat2) / 2, (lon1 + lon2) / 2],
        zoom_start=7,
        tiles="OpenStreetMap"
    )

    folium.Marker(
        [lat1, lon1],
        tooltip="Start Point",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

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
                    "weight": 6,
                }
            },
            "polygon": False,
            "circle": False,
            "rectangle": False,
            "marker": False,
            "circlemarker": False,
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
        st.info("Gambar rute bebas mengikuti jalur laut (tidak harus 5 titik)")
        return None

    geom = drawing.get("geometry", {})

    if geom.get("type") != "LineString":
        st.warning("Objek harus berupa polyline.")
        return None

    coords = geom.get("coordinates", [])

    # =========================
    # VALIDASI MINIMAL
    # =========================
    if len(coords) < 2:
        st.error("Rute minimal 2 titik.")
        return None

    # 🔥 POLYLINE ASLI (TIDAK DIUBAH)
    full_route = [(pt[1], pt[0]) for pt in coords]

    # 🔥 5 TITIK UNTUK 4 SEGMENT
    titik5 = split_route_into_4_segments(full_route)

    # =========================
    # MAP FINAL (SAMA DENGAN GAMBAR USER)
    # =========================
    m2 = folium.Map(
        location=[(lat1 + lat2) / 2, (lon1 + lon2) / 2],
        zoom_start=7,
        tiles="OpenStreetMap"
    )

    folium.PolyLine(
        locations=full_route,
        color="#1565C0",
        weight=6,
    ).add_to(m2)

    for i, (lat, lon) in enumerate(full_route, start=1):
        numbered_marker(lat, lon, i).add_to(m2)

    st.success("✅ Rute valid & tersimpan (persis seperti gambar)")

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

        # 🔥 untuk analisis (4 segmen)
        "titik5": titik5,

        # 🔥 untuk visual & akurasi rute
        "polyline_full": full_route
    }
