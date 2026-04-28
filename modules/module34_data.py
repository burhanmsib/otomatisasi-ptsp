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
def open_dataset_with_retry(url, max_try=3, delay=2):
    for i in range(max_try):
        try:
            return xr.open_dataset(url)
        except:
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
    if raw is None or str(raw).strip() == "":
        return None

    s = str(raw)
    s = re.sub(r"\d{1,2}[.:]\d{2}(-\d{1,2}[.:]\d{2})?", "", s)
    s = s.replace("/", " ")

    month_map = {
        "Januari":"January","Februari":"February","Maret":"March",
        "April":"April","Mei":"May","Juni":"June","Juli":"July",
        "Agustus":"August","September":"September",
        "Oktober":"October","November":"November","Desember":"December"
    }

    for indo, eng in month_map.items():
        s = s.replace(indo, eng)

    try:
        return parser.parse(s, dayfirst=True)
    except:
        return None


# =========================
# 🔥 POLYGON SAMPLING
# =========================
def generate_polygon_sampling_points(pointA, pointB, buffer_deg=0.2, grid_size=4):

    lat1, lon1 = pointA
    lat2, lon2 = pointB

    line = LineString([(lon1, lat1), (lon2, lat2)])
    polygon = line.buffer(buffer_deg)

    minx, miny, maxx, maxy = polygon.bounds

    points = []

    for lat in np.linspace(miny, maxy, grid_size):
        for lon in np.linspace(minx, maxx, grid_size):
            p = Point(lon, lat)
            if polygon.contains(p):
                points.append((lat, lon))

    if not points:
        return [pointA, pointB]

    return points

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
        rain = s.get("rain", {}).get("precip")
        label = classify_weather_from_rain(rain)
        labels.append(label)

    order = ["Clear", "Slight Rain", "Moderate Rain", "Heavy Rain"]

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

    ds_wave = open_dataset_with_retry(
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/ww3gfs/{YYYY}/{MM}/w3g_hires_{YYYY}{MM}{DD}_1200.nc"
    )

    ds_cur = open_dataset_with_retry(
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/fvcom/{YYYY}/{MM}/InaFlows_{YYYY}{MM}{DD}_1200.nc"
    )

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
# WEATHER EXTRACTION
# =========================
def extract_hourly_weather(ds_wave, ds_cur, ds_rain, t, lat, lon):

    rain_val = None

    if ds_rain is not None:
        try:
            var = list(ds_rain.data_vars)[0]
            da = ds_rain[var]

            if "time" in da.dims:
                da = da.sel(time=t, method="nearest")

            lat_idx = np.abs(da["lat"].values - lat).argmin()
            lon_idx = np.abs(da["lon"].values - lon).argmin()

            rain_val = float(da.isel(lat=lat_idx, lon=lon_idx).values)

        except:
            rain_val = None

    return {
        "wave": {
            "hs": safe_extract(ds_wave,"hs",t,lat,lon),
        },
        "wind": {
            "u": safe_extract(ds_wave,"uwnd",t,lat,lon),
            "v": safe_extract(ds_wave,"vwnd",t,lat,lon)
        },
        "current": {
            "u": safe_extract(ds_cur,"u",t,lat,lon),
            "v": safe_extract(ds_cur,"v",t,lat,lon)
        },
        "rain": {
            "precip": rain_val
        }
    }


# =========================
# MAIN PROCESS (POLYGON FINAL - COMPATIBLE)
# =========================
def process_module34(row, polyline, tz="WIB", ds_wave=None, ds_cur=None, ds_rain=None):

    dt_local = normalize_date(row["Tanggal Koordinat"])
    if dt_local is None:
        return None

    tz_offset = TZ_OFFSET.get(tz, 7)

    dt_utc0 = dt_local.replace(
        tzinfo=timezone(timedelta(hours=tz_offset))
    ).astimezone(timezone.utc).replace(tzinfo=None)

    route = [(p[0], p[1]) for p in polyline]

    segments = []
    n = len(route)

    for i in range(4):

        t0 = dt_utc0 + timedelta(hours=i * 6)

        start_idx = int(i * (n-1) / 4)
        end_idx   = int((i+1) * (n-1) / 4) + 1

        segment_route = route[start_idx:end_idx]

        if len(segment_route) < 2:
            segment_route = route

        # =========================
        # 🔥 POLYGON SAMPLING (GANTI LINE)
        # =========================
        pointA = segment_route[0]
        pointB = segment_route[-1]

        sample_points = generate_polygon_sampling_points(
            pointA, pointB,
            buffer_deg=0.2,   # ±20–25 km
            grid_size=4       # max ~10 titik
        )

        samples = []

        for j, (lat, lon) in enumerate(sample_points):

            # 🔥 tetap pakai timestep lama (3 jam)
            t = t0 + timedelta(hours=j * 3)

            sample = extract_hourly_weather(
                ds_wave, ds_cur, ds_rain,
                t, lat, lon
            )
            samples.append(sample)

        # =========================
        # 🔥 WEATHER RANGE (TETAP)
        # =========================
        weather = build_weather_range(samples)

        segments.append({
            "interval": f"T{i*6}-T{(i+1)*6}",
            "samples": samples,
            "weather": weather
        })

    return {
        "tanggal": dt_local,
        "tz": tz,
        "segments": segments
    }
