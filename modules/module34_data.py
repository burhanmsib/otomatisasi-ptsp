# =========================
# MODULE 3 + 4 (POLYGON VERSION - FINAL)
# =========================

import re
import numpy as np
import xarray as xr
import streamlit as st
import ftplib
import tempfile
import os
import time

from datetime import datetime, timedelta, timezone
from dateutil import parser
from shapely.geometry import LineString, Point

# =========================
# TIMEOUT
# =========================
os.environ["OPENDAP_TIMEOUT"] = "60"

# =========================
# RETRY
# =========================
def open_dataset_with_retry(
    url,
    max_try=3,
    delay=2
):

    for i in range(max_try):

        try:

            ds = xr.open_dataset(
                url,
                engine="netcdf4"
            )

            return ds

        except Exception as e:

            print(
                f"[Retry {i+1}] gagal buka: {url}"
            )

            print(e)

            time.sleep(delay)

    return None


# =========================
# CONSTANT
# =========================
TZ_OFFSET = {
    "WIB": 7,
    "WITA": 8,
    "WIT": 9
}


# =========================
# DATE NORMALIZATION
# =========================
def normalize_date(raw):

    if raw is None:
        return None

    s = str(raw).strip()

    if not s:
        return None

    # =========================
    # FORMAT STANDAR PTSP
    # YYYY-MM-DD
    # =========================

    try:
        return datetime.strptime(
            s,
            "%Y-%m-%d"
        )
    except:
        pass

    # =========================
    # YYYY/MM/DD
    # =========================

    try:
        return datetime.strptime(
            s,
            "%Y/%m/%d"
        )
    except:
        pass

    # =========================
    # DD-MM-YYYY
    # =========================

    try:
        return datetime.strptime(
            s,
            "%d-%m-%Y"
        )
    except:
        pass

    # =========================
    # DD/MM/YYYY
    # =========================

    try:
        return datetime.strptime(
            s,
            "%d/%m/%Y"
        )
    except:
        pass

    # =========================
    # BULAN INDONESIA
    # =========================

    month_map = {
        "Januari":"January",
        "Februari":"February",
        "Maret":"March",
        "April":"April",
        "Mei":"May",
        "Juni":"June",
        "Juli":"July",
        "Agustus":"August",
        "September":"September",
        "Oktober":"October",
        "November":"November",
        "Desember":"December"
    }

    clean = s

    for indo, eng in month_map.items():
        clean = clean.replace(
            indo,
            eng
        )

    # =========================
    # DD Month YYYY
    # =========================

    try:
        return datetime.strptime(
            clean,
            "%d %B %Y"
        )
    except:
        pass

    # =========================
    # FALLBACK TERAKHIR
    # =========================

    try:
        return parser.parse(
            clean,
            dayfirst=True
        )
    except:
        return None

# =========================
# 🔥 SPLIT POLYLINE
# SUPPORT:
# - titik tunggal
# - polyline
# =========================
def split_polyline_into_segments(
    full_route,
    n_segments=4
):

    # =========================
    # TIDAK ADA TITIK
    # =========================
    if not full_route:
        return []

    # =========================
    # MODE TITIK
    # =========================
    if len(full_route) < 2:

        # 🔥 tetap 4 segmen
        return [
            full_route
            for _ in range(n_segments)
        ]

    try:

        line = LineString([
            (lon, lat)
            for lat, lon in full_route
        ])

        # 🔥 geometry invalid
        if line.is_empty or not line.is_valid:

            return [
                full_route
                for _ in range(n_segments)
            ]

        segments = []

        for i in range(n_segments):

            start_f = i / n_segments
            end_f = (i + 1) / n_segments

            start_d = line.length * start_f
            end_d = line.length * end_f

            coords = []

            for d in np.linspace(
                start_d,
                end_d,
                10
            ):

                p = line.interpolate(d)

                coords.append((
                    p.y,
                    p.x
                ))

            segments.append(coords)

        return segments

    except:

        # 🔥 fallback aman
        return [
            full_route
            for _ in range(n_segments)
        ]

# =========================
# 🔥 POLYGON SAMPLING
# SUPPORT:
# - titik tunggal
# - polyline/rute
# =========================
def generate_polygon_sampling_points(
    segment_route,
    route_buffer=0.12
):

    # =========================
    # VALIDASI TITIK
    # =========================
    valid_points = []

    for p in segment_route:

        if (
            isinstance(p, (list, tuple))
            and len(p) == 2
        ):

            lat, lon = p

            if (
                lat is not None
                and lon is not None
            ):
                valid_points.append((lat, lon))

    # =========================
    # TIDAK ADA TITIK
    # =========================
    if not valid_points:
        return []

    # =========================
    # MODE TITIK
    # =========================
    if len(valid_points) == 1:

        lat, lon = valid_points[0]

        # 🔥 buffer titik lebih kecil
        point_buffer = 0.05

        points = []

        for dlat in np.linspace(
            -point_buffer,
            point_buffer,
            3
        ):

            for dlon in np.linspace(
                -point_buffer,
                point_buffer,
                3
            ):

                points.append((
                    lat + dlat,
                    lon + dlon
                ))

        return points

    # =========================
    # MODE RUTE
    # =========================
    try:

        line = LineString([
            (lon, lat)
            for lat, lon in valid_points
        ])

        # 🔥 geometry invalid
        if line.is_empty or not line.is_valid:
            return valid_points

        polygon = line.buffer(route_buffer)

        minx, miny, maxx, maxy = polygon.bounds

        points = []

        for lat in np.linspace(miny, maxy, 3):

            for lon in np.linspace(minx, maxx, 3):

                p = Point(lon, lat)

                if polygon.contains(p):
                    points.append((lat, lon))

        # 🔥 fallback
        if not points:
            return valid_points

        return points

    except:

        # 🔥 fallback aman
        return valid_points
        
# =========================
# 🔥 WEATHER RANGE BUILDER
# =========================
def classify_weather_from_rain(rain):

    if rain is None:
        return "Clear"

    if rain < 0.5:
        return "Clear"
    elif rain < 5:
        return "Slight Rain"
    elif rain < 10:
        return "Moderate Rain"
    else:
        return "Heavy Rain"


def build_weather_range(samples):

    labels = []

    for s in samples:

        # 🔥 skip kalau bukan dict
        if not isinstance(s, dict):
            continue

        rain = s.get("rain", {}).get("precip")

        label = classify_weather_from_rain(rain)

        labels.append(label)

    order = [
        "Clear",
        "Slight Rain",
        "Moderate Rain",
        "Heavy Rain"
    ]

    valid = [l for l in labels if l in order]

    if not valid:
        return "Clear"

    min_idx = min(order.index(l) for l in valid)
    max_idx = max(order.index(l) for l in valid)

    if min_idx == max_idx:
        return order[min_idx]

    return f"{order[min_idx]} to {order[max_idx]}"


# =========================
# GSMAP (RAIN)
# =========================
@st.cache_resource(ttl=3600)
def load_gsmap_cached(dt):
    try:
        ftp_host = st.secrets["ftp"]["host"]
        ftp_user = st.secrets["ftp"]["user"]
        ftp_pass = st.secrets["ftp"]["pass"]

        Y, M, D, H = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"), dt.strftime("%H")
        remote_path = f"/himawari6/GSMaP/netcdf/{Y}/{M}/{D}/GSMaP_{Y}{M}{D}{H}00.nc"

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
        tmp_path = tmp.name
        tmp.close()

        ftp = ftplib.FTP(ftp_host, timeout=20)
        ftp.login(ftp_user, ftp_pass)

        with open(tmp_path, "wb") as f:
            ftp.retrbinary(f"RETR {remote_path}", f.write)

        ftp.quit()

        ds = xr.open_dataset(tmp_path)
        os.remove(tmp_path)

        return ds

    except:
        return None


# =========================
# LOAD DATASET
# =========================
@st.cache_resource(ttl=3600)
def load_datasets_cached(dt_input):

    dt = normalize_date(dt_input)
    if dt is None:
        return None, None, None

    user = st.secrets["bmkg"]["user"]
    password = st.secrets["bmkg"]["pass"]

    YYYY, MM, DD = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")

    # =========================
    # WAVE
    # PRIORITAS:
    # 1. 1200
    # 2. 0000
    # =========================
    ds_wave = None
    
    wave_urls = [
    
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/ww3gfs/{YYYY}/{MM}/w3g_hires_{YYYY}{MM}{DD}_1200.nc",
    
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/ww3gfs/{YYYY}/{MM}/w3g_hires_{YYYY}{MM}{DD}_0000.nc",
    ]
    
    for url in wave_urls:
    
        ds_wave = open_dataset_with_retry(url)
    
        if ds_wave is not None:
            print(f"WAVE loaded: {url}")
            break
        
    # =========================
    # CURRENT
    # PRIORITAS:
    # 1. 1200
    # 2. 0000
    # =========================
    ds_cur = None
    
    current_urls = [
    
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/fvcom/{YYYY}/{MM}/InaFlows_{YYYY}{MM}{DD}_1200.nc",
    
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/fvcom/{YYYY}/{MM}/InaFlows_{YYYY}{MM}{DD}_0000.nc",
    ]
    
    for url in current_urls:
    
        ds_cur = open_dataset_with_retry(url)
    
        if ds_cur is not None:
            print(f"CURRENT loaded: {url}")
            break
    ds_rain = load_gsmap_cached(dt)

    return ds_wave, ds_cur, ds_rain


# =========================
# SAFE EXTRACT
# =========================
def safe_extract(ds, var, t, lat, lon, depth=None):

    if ds is None or var not in ds:
        return 0.0

    try:
        da = ds[var]

        if "time" in da.dims:
            da = da.sel(time=t, method="nearest")

        if depth is not None and "depth" in da.dims:
            da = da.sel(depth=0, method="nearest")

        return float(da.sel(lat=lat, lon=lon, method="nearest").values)

    except:
        return 0.0

# =========================
# WIND SPEED
# =========================
def wind_speed(u, v):

    if u is None or v is None:
        return 0.0

    try:
        # 🔥 kemungkinan dataset sudah knot
        spd = np.sqrt(u**2 + v**2)

        return float(spd)

    except:
        return 0.0

# =========================
# WEATHER EXTRACTION (FINAL FIX - STABLE)
# =========================
def extract_hourly_weather(ds_wave, ds_cur, ds_rain, t, lat, lon):

    try:

        # =========================
        # GSMAP SAFE EXTRACT
        # =========================
        rain_val = None

        if ds_rain is not None:

            try:
                var = list(ds_rain.data_vars)[0]
                da = ds_rain[var]

                if "time" in da.dims:
                    da = da.sel(time=t, method="nearest")

                lat_name = "lat" if "lat" in da.coords else "latitude"
                lon_name = "lon" if "lon" in da.coords else "longitude"

                lat_vals = da[lat_name].values
                lon_vals = da[lon_name].values

                lat_idx = np.abs(lat_vals - lat).argmin()
                lon_idx = np.abs(lon_vals - lon).argmin()

                rain_val = float(
                    da.isel({
                        lat_name: lat_idx,
                        lon_name: lon_idx
                    }).values
                )

                if np.isnan(rain_val):
                    rain_val = None

            except:
                rain_val = None

        # =========================
        # WIND
        # =========================
        u_wind = safe_extract(ds_wave, "uwnd", t, lat, lon)
        v_wind = safe_extract(ds_wave, "vwnd", t, lat, lon)

        wind_knot = wind_speed(u_wind, v_wind)

        # =========================
        # RETURN
        # =========================
        return {

            "wave": {
                "hs": safe_extract(
                    ds_wave,
                    "hs",
                    t,
                    lat,
                    lon
                ),
            },

            "wind": {
                "u": u_wind,
                "v": v_wind,
                "speed_knot": wind_knot
            },

            "current": {
                "u": safe_extract(
                    ds_cur,
                    "u",
                    t,
                    lat,
                    lon,
                    depth=0.5
                ),

                "v": safe_extract(
                    ds_cur,
                    "v",
                    t,
                    lat,
                    lon,
                    depth=0.5
                )
            },

            "rain": {
                "precip": rain_val
            }
        }

    except Exception:

        # 🔥 RETURN AMAN
        return {
            "wave": {
                "hs": None
            },

            "wind": {
                "u": None,
                "v": None,
                "speed_knot": None
            },

            "current": {
                "u": None,
                "v": None
            },

            "rain": {
                "precip": None
            }
        }

# =========================
# MAIN PROCESS (FINAL FIX)
# =========================
def process_module34(row, polyline, tz="WIB", ds_wave=None, ds_cur=None, ds_rain=None):

    # dt_local = normalize_date(row["Tanggal Koordinat"])

    dt_local = normalize_date(
        row["Tanggal Koordinat"]
    )
    
    if dt_local is None:
        return None

    tz_offset = TZ_OFFSET.get(tz, 7)

    dt_utc0 = dt_local.replace(
        tzinfo=timezone(timedelta(hours=tz_offset))
    ).astimezone(timezone.utc).replace(tzinfo=None)

    route = [
        (p[0], p[1])
        for p in polyline
        if (
            isinstance(p, (list, tuple))
            and len(p) >= 2
        )
    ]

    # =========================
    # DETEKSI MODE
    # =========================
    is_point_mode = len(route) < 2
    
    # 🔥 kalau route kosong
    if not route:
        return None

    # =========================
    # MODE TITIK
    # =========================
    if is_point_mode:
    
        segments_route = [route]
    
    # =========================
    # MODE RUTE
    # =========================
    else:
    
        segments_route = split_polyline_into_segments(
            route,
            4
        )

    if not segments_route:

        segments_route = [[route[0]]] * 4

    while len(segments_route) < 4:

        segments_route.append(
            segments_route[-1]
        )
    
    segments = []
    
    for i in range(len(segments_route)):
    
        t0 = dt_utc0 + timedelta(
            hours=i * 6
        )
    
        segment_route = segments_route[i]

        # 🔥 polygon mengikuti bentuk segmen
        sample_points = generate_polygon_sampling_points(segment_route)

        times = [
            t0,
            t0 + timedelta(hours=3),
            t0 + timedelta(hours=6)
        ]

        samples = []

        for t in times:
            t = t.replace(minute=0, second=0)

            for lat, lon in sample_points:

                sample = extract_hourly_weather(
                    ds_wave, ds_cur, ds_rain,
                    t, lat, lon
                )
                if isinstance(sample, dict):
                    samples.append(sample)

        weather = build_weather_range(samples)

        segments.append({
            "interval": (
                f"T{i*6}-T{(i+1)*6}"
                if not is_point_mode
                else "T0-T24"
            ),
            "samples": samples,
            "weather": weather
        })

    return {
        "tanggal": dt_local,
        "tz": tz,
        "segments": segments
    }
