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
# BUILD PREVIEW MAP
# =========================
def build_route_preview_map(
    full_route,
    titik5,
    lat1,
    lon1,
    lat2,
    lon2
):

    m2 = folium.Map(
        location=[
            (lat1 + lat2) / 2,
            (lon1 + lon2) / 2
        ],
        zoom_start=6,
        tiles="OpenStreetMap"
    )

    # GARIS RUTE
    folium.PolyLine(
        locations=full_route,
        color="#1565C0",
        weight=6
    ).add_to(m2)

    # START
    folium.Marker(
        [lat1, lon1],
        tooltip="Start Point",
        icon=folium.Icon(
            color="green",
            icon="play"
        )
    ).add_to(m2)

    # END
    folium.Marker(
        [lat2, lon2],
        tooltip="End Point",
        icon=folium.Icon(
            color="red",
            icon="flag"
        )
    ).add_to(m2)

    # TITIK 1–5
    for i, (lat, lon) in enumerate(
        titik5,
        start=1
    ):

        folium.Marker(
            [lat, lon],
            tooltip=f"Titik {i}",
            icon=folium.DivIcon(
                html=f"""
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
                """
            )
        ).add_to(m2)

    return m2

# =========================
# MAIN FUNCTION
# =========================
def process_route_segment_module2_streamlit(row, map_key, saved_route=None):

    st.subheader("Mode Input Lokasi")
    
    mode = st.radio(
        "Pilih Mode",
        ["Gambar Rute", "Titik Tunggal"],
        horizontal=True,
        key=f"mode_{map_key}"
    )

    # =========================
    # VALIDASI SAVED ROUTE
    # =========================
    if saved_route is not None:
    
        saved_mode = saved_route.get("mode")
    
        # Data lama sebelum field "mode" ditambahkan
        if saved_mode is None:
    
            saved_points = saved_route.get(
                "polyline_full",
                []
            )
    
            if len(saved_points) >= 2:
                saved_mode = "Gambar Rute"
            else:
                saved_mode = "Titik Tunggal"
    
        # Jangan tampilkan data dari mode berbeda
        if saved_mode != mode:
            saved_route = None

    # =========================
    # MODE TITIK TUNGGAL
    # =========================
    if mode == "Titik Tunggal":
    
        # =========================
        # BACA KOORDINAT DARI DATA
        # =========================
        lat, lon = parse_decimal_coordinate(
            row.get("Koordinat Awal (Desimal)")
        )
    
        if lat is None or lon is None:
            st.error(
                "Koordinat titik tidak tersedia atau format tidak valid."
            )
            return saved_route
    
        # =========================
        # INFO KOORDINAT
        # =========================
        st.info(
            f"📍 Koordinat lokasi: "
            f"{lat:.5f}, {lon:.5f}"
        )
    
        # =========================
        # SIMPAN TITIK
        # =========================
        if st.button(
            "💾 Simpan Titik",
            key=f"btn_point_{map_key}"
        ):
    
            # Zona waktu otomatis
            tz = get_timezone(lat, lon)
    
            # =========================
            # GEOJSON
            # =========================
            geojson = json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {
                                "name": "Titik Analisis"
                            },
                            "geometry": {
                                "type": "Point",
                                "coordinates": [
                                    lon,
                                    lat
                                ]
                            }
                        }
                    ]
                },
                indent=2
            )
    
            # =========================
            # BUAT PETA
            # =========================
            point_map = folium.Map(
                location=[lat, lon],
                zoom_start=8,
                tiles="OpenStreetMap"
            )
    
            folium.Marker(
                [lat, lon],
                tooltip="Titik Analisis",
                icon=folium.Icon(
                    color="red",
                    icon="map-marker"
                )
            ).add_to(point_map)
    
            # Simpan HTML
            html = point_map.get_root().render()
    
            # =========================
            # SIMPAN HASIL
            # =========================
            saved_route = {
                "mode": "Titik Tunggal",
                "tanggal": row.get(
                    "Tanggal Koordinat"
                ),
                "awal": (lat, lon),
                "akhir": (lat, lon),
                "titik5": [(lat, lon)],
                "polyline_full": [(lat, lon)],
                "tz": tz,
                "geojson": geojson,
                "map_html": html
            }
    
        # =========================
        # PREVIEW TITIK TERSIMPAN
        # =========================
        if saved_route is not None:
    
            saved_points = saved_route.get(
                "titik5",
                []
            )
    
            if saved_points:
    
                point_lat, point_lon = (
                    saved_points[0]
                )
    
                tz = saved_route.get(
                    "tz",
                    "-"
                )
    
                st.success(
                    f"✅ Titik tersimpan: "
                    f"{point_lat:.5f}, "
                    f"{point_lon:.5f} "
                    f"({tz})"
                )
    
                # =========================
                # PREVIEW MAP
                # =========================
                preview_map = folium.Map(
                    location=[
                        point_lat,
                        point_lon
                    ],
                    zoom_start=8,
                    tiles="OpenStreetMap"
                )
    
                folium.Marker(
                    [
                        point_lat,
                        point_lon
                    ],
                    tooltip="Titik Analisis",
                    icon=folium.Icon(
                        color="red",
                        icon="map-marker"
                    )
                ).add_to(preview_map)
    
                st.markdown(
                    "### 📍 Preview Titik"
                )
    
                st_folium(
                    preview_map,
                    height=500,
                    use_container_width=True,
                    key=f"point_preview_{map_key}"
                )
    
                # =========================
                # DOWNLOAD
                # =========================
                col1, col2 = st.columns(2)
    
                with col1:
    
                    if saved_route.get("geojson"):
    
                        st.download_button(
                            "📍 Download Titik (.geojson)",
                            data=saved_route["geojson"],
                            file_name=f"point_{map_key}.geojson",
                            mime="application/geo+json",
                            key=f"point_geojson_{map_key}"
                        )
    
                with col2:
    
                    if saved_route.get("map_html"):
    
                        st.download_button(
                            "🌐 Download Peta Titik (.html)",
                            data=saved_route["map_html"],
                            file_name=f"point_{map_key}.html",
                            mime="text/html",
                            key=f"point_html_{map_key}"
                        )
    
        return saved_route
    
    # =========================
    # MODE GAMBAR RUTE
    # =========================
    lat1, lon1 = parse_decimal_coordinate(
        row.get("Koordinat Awal (Desimal)")
    )
    
    lat2, lon2 = parse_decimal_coordinate(
        row.get("Koordinat Akhir (Desimal)")
    )
    
    if None in (lat1, lon1, lat2, lon2):
        st.error("Format koordinat tidak valid")
        return saved_route
    
    
    # =========================
    # LAYOUT
    # =========================
    col_draw, col_preview = st.columns([1, 1])
    
    
    # =========================
    # MAP DRAW
    # =========================
    m = folium.Map(
        location=[
            (lat1 + lat2) / 2,
            (lon1 + lon2) / 2
        ],
        zoom_start=6,
        tiles="OpenStreetMap"
    )
    
    
    # START
    folium.Marker(
        [lat1, lon1],
        tooltip="Start Point",
        icon=folium.Icon(
            color="green",
            icon="play"
        )
    ).add_to(m)
    
    
    # END
    folium.Marker(
        [lat2, lon2],
        tooltip="End Point",
        icon=folium.Icon(
            color="red",
            icon="flag"
        )
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
    
    
    # =========================
    # GAMBAR RUTE
    # =========================
    with col_draw:
    
        st.markdown("### 🗺️ Gambar Rute")
    
        output = st_folium(
            m,
            height=800,
            use_container_width=True,
            key=f"draw_map_{map_key}",
            returned_objects=[
                "last_active_drawing"
            ]
        )
    
    
    drawing = output.get(
        "last_active_drawing"
    )
    
    
    # ==================================================
    # DEFAULT:
    # GUNAKAN RUTE YANG SUDAH TERSIMPAN
    # ==================================================
    route_result = saved_route
    
    
    # ==================================================
    # JIKA ADA GAMBAR BARU
    # PROSES DAN TIMPA RUTE LAMA
    # ==================================================
    if drawing is not None:
    
        geom = drawing.get(
            "geometry",
            {}
        )
    
        if geom.get("type") == "LineString":
    
            coords = geom.get(
                "coordinates",
                []
            )
    
            if len(coords) >= 2:
    
                # POLYLINE ASLI
                full_route = [
                    (pt[1], pt[0])
                    for pt in coords
                ]
    
                # 5 TITIK SEGMENT
                titik5 = (
                    split_route_into_4_segments(
                        full_route
                    )
                )

                # ZONA WAKTU BERDASARKAN TITIK REPRESENTATIF RUTE
                rep_lat, rep_lon = titik5[2]
                tz = get_timezone(rep_lat, rep_lon)
    
                # BUILD PREVIEW MAP
                m2 = build_route_preview_map(
                    full_route,
                    titik5,
                    lat1,
                    lon1,
                    lat2,
                    lon2
                )
    
                # GEOJSON
                geojson = create_geojson(
                    full_route,
                    titik5
                )
    
                # HTML
                html = (
                    m2
                    .get_root()
                    .render()
                )
    
                # HASIL BARU
                route_result = {
    
                    "tanggal":
                    row.get(
                        "Tanggal Koordinat"
                    ),
    
                    "awal":
                    (lat1, lon1),
    
                    "akhir":
                    (lat2, lon2),
    
                    "titik5":
                    titik5,
    
                    "polyline_full":
                    full_route,
    
                    "geojson":
                    geojson,
    
                    "map_html":
                    html,

                    "tz": tz
                }
    
    
    # ==================================================
    # PREVIEW RUTE
    # ==================================================
    with col_preview:
    
        st.markdown(
            "### 📍 Preview Rute"
        )
    
        # =========================
        # BELUM ADA RUTE
        # =========================
        if route_result is None:
    
            st.info(
                "Belum ada rute yang digambar."
            )
    
        # =========================
        # SUDAH ADA RUTE
        # =========================
        else:
    
            full_route = (
                route_result.get(
                    "polyline_full",
                    []
                )
            )
    
            titik5 = (
                route_result.get(
                    "titik5",
                    []
                )
            )
    
            if full_route and titik5:
    
                # Gunakan koordinat tersimpan
                awal = route_result.get(
                    "awal",
                    (lat1, lon1)
                )
    
                akhir = route_result.get(
                    "akhir",
                    (lat2, lon2)
                )
    
                preview_lat1, preview_lon1 = awal
                preview_lat2, preview_lon2 = akhir
    
                # BUILD ULANG PETA
                preview_map = (
                    build_route_preview_map(
                        full_route,
                        titik5,
                        preview_lat1,
                        preview_lon1,
                        preview_lat2,
                        preview_lon2
                    )
                )
    
                st_folium(
                    preview_map,
                    height=800,
                    use_container_width=True,
                    key=f"final_map_{map_key}"
                )
    
    
    # ==================================================
    # DOWNLOAD RUTE TERSIMPAN
    # ==================================================
    if route_result is not None:
    
        st.success(
            "Rute tersimpan"
        )
    
        col_download1, col_download2 = (
            st.columns(2)
        )
    
    
        # =========================
        # GEOJSON
        # =========================
        with col_download1:
    
            geojson_data = (
                route_result.get(
                    "geojson"
                )
            )
    
            # Untuk data lama yang belum punya geojson
            if (
                not geojson_data
                and route_result.get(
                    "polyline_full"
                )
                and route_result.get(
                    "titik5"
                )
            ):
    
                geojson_data = (
                    create_geojson(
                        route_result[
                            "polyline_full"
                        ],
                        route_result[
                            "titik5"
                        ]
                    )
                )
    
                route_result[
                    "geojson"
                ] = geojson_data
    
    
            if geojson_data:
    
                st.download_button(
                    label=(
                        "📍 Download Route "
                        "(.geojson)"
                    ),
                    data=geojson_data,
                    file_name=(
                        f"route_{map_key}.geojson"
                    ),
                    mime=(
                        "application/geo+json"
                    ),
                    key=(
                        f"download_geojson_"
                        f"{map_key}"
                    )
                )
    
    
        # =========================
        # HTML
        # =========================
        with col_download2:
    
            html_data = (
                route_result.get(
                    "map_html"
                )
            )
    
            # Untuk data lama yang belum punya HTML
            if (
                not html_data
                and route_result.get(
                    "polyline_full"
                )
                and route_result.get(
                    "titik5"
                )
            ):
    
                awal = route_result.get(
                    "awal",
                    (lat1, lon1)
                )
    
                akhir = route_result.get(
                    "akhir",
                    (lat2, lon2)
                )
    
                html_map = (
                    build_route_preview_map(
                        route_result[
                            "polyline_full"
                        ],
                        route_result[
                            "titik5"
                        ],
                        awal[0],
                        awal[1],
                        akhir[0],
                        akhir[1]
                    )
                )
    
                html_data = (
                    html_map
                    .get_root()
                    .render()
                )
    
                route_result[
                    "map_html"
                ] = html_data
    
    
            if html_data:
    
                st.download_button(
                    label=(
                        "🌐 Download Route Map "
                        "(.html)"
                    ),
                    data=html_data,
                    file_name=(
                        f"route_{map_key}.html"
                    ),
                    mime="text/html",
                    key=(
                        f"download_html_"
                        f"{map_key}"
                    )
                )
    
    
    return route_result
