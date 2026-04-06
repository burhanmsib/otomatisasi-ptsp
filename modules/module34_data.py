# =========================
# MODULE 3 + 4 (FINAL COMPLETE - SUPER FAST CURRENT OPTIMIZED)
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
from shapely.geometry import LineString


# =========================
# 🔥 TAMBAHAN: TIMEOUT
# =========================
os.environ["OPENDAP_TIMEOUT"] = "60"


# =========================
# 🔥 TAMBAHAN: RETRY FUNCTION
# =========================
def open_dataset_with_retry(url, max_try=3, delay=2):
    for i in range(max_try):
        try:
            ds = xr.open_dataset(url)
            return ds
        except Exception:
            print(f"[Retry {i+1}] gagal buka: {url}")
            time.sleep(delay)
    return None


# =========================
# CONSTANTS
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

    s = s.strip()

    formats = [
        "%d.%m.%Y", "%d-%m-%Y", "%d %B %Y",
        "%Y-%m-%d", "%d %b %Y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except:
            continue

    try:
        return parser.parse(s, dayfirst=True)
    except:
        return None


# =========================
# ROUTE SAMPLING
# =========================
def generate_3_points_along_route(polyline):
    if not polyline or len(polyline) < 2:
        return polyline

    line = LineString([(lon, lat) for lat, lon in polyline])

    fractions = [0.0, 0.5, 1.0]
    points = []

    for f in fractions:
        p = line.interpolate(f, normalized=True)
        points.append((p.y, p.x))

    return points


# =========================
# WEATHER CLASSIFICATION
# =========================
def classify_weather_from_rain(rain):
    if rain is None:
        return "Unknown"

    if rain < 0.5:
        return "Clear"
    elif rain < 5:
        return "Slight Rain"
    elif rain < 10:
        return "Moderate Rain"
    else:
        return "Heavy Rain"


# =========================
# WEATHER RANGE
# =========================
def build_weather_range(samples):
    labels = []
    for s in samples:
        rain = s["rain"]["precip"]
        label = classify_weather_from_rain(rain)
        labels.append(label)

    order = ["Clear", "Slight Rain", "Moderate Rain", "Heavy Rain"]
    labels = [l for l in labels if l in order]

    if not labels:
        return "Clear"

    min_idx = min(order.index(l) for l in labels)
    max_idx = max(order.index(l) for l in labels)

    if min_idx == max_idx:
        return order[min_idx]

    return f"{order[min_idx]} to {order[max_idx]}"


# =========================
# GSMAP (CACHE)
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

    except Exception as e:
        st.warning(f"GSMAP gagal load: {e}")
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

    ds_wave = None
    for url in [
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/ww3gfs/{YYYY}/{MM}/w3g_hires_{YYYY}{MM}{DD}_1200.nc",
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/ww3gfs/{YYYY}/{MM}/w3g_hires_{YYYY}{MM}{DD}_0000.nc",
    ]:
        ds_wave = open_dataset_with_retry(url)
        if ds_wave is not None:
            break

    ds_cur = None
    for url in [
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/fvcom/{YYYY}/{MM}/InaFlows_{YYYY}{MM}{DD}_1200.nc",
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/fvcom/{YYYY}/{MM}/InaFlows_{YYYY}{MM}{DD}_0000.nc",
    ]:
        ds_cur = open_dataset_with_retry(url)
        if ds_cur is not None:
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
# 🔥 SMART CURRENT FINAL (SUPER FAST OPTIMIZED)
# =========================
def get_current_smart(ds_cur, t, lat, lon, max_window=2):
    try:
        # 1. Pilih waktu terdekat
        da_u = ds_cur["u"].sel(time=t, method="nearest")
        da_v = ds_cur["v"].sel(time=t, method="nearest")
        
        # 2. Cari index latitude dan longitude terdekat
        lat_vals = da_u["lat"].values
        lon_vals = da_u["lon"].values
        
        lat_idx = np.abs(lat_vals - lat).argmin()
        lon_idx = np.abs(lon_vals - lon).argmin()
        
        # 3. BATCH FETCHING: Tarik sekotak area (misal 5x5) SEKALIGUS
        lat_slice = slice(max(0, lat_idx - max_window), lat_idx + max_window + 1)
        lon_slice = slice(max(0, lon_idx - max_window), lon_idx + max_window + 1)
        
        u_grid = da_u.isel(lat=lat_slice, lon=lon_slice).values
        v_grid = da_v.isel(lat=lat_slice, lon=lon_slice).values
        
        # 4. Tentukan posisi titik utama di dalam kotak grid yang ditarik
        center_i = min(lat_idx, max_window) 
        center_j = min(lon_idx, max_window)
        
        # 5. Cek titik utama terlebih dahulu
        u_center, v_center = u_grid[center_i, center_j], v_grid[center_i, center_j]
        if not np.isnan(u_center) and not np.isnan(v_center):
            return float(u_center), float(v_center)
        
        # 6. Jika titik utama NaN (Kena daratan), hitung semua titik di grid
        spd_grid = np.hypot(u_grid, v_grid)
        
        # Cek apakah seluruh area kotak ini NaN (daratan total)
        if np.all(np.isnan(spd_grid)):
            return None, None
            
        # 7. Ambil kecepatan arus tertinggi di area sekitar titik utama
        max_idx = np.unravel_index(np.nanargmax(spd_grid), spd_grid.shape)
        u_best = u_grid[max_idx]
        v_best = v_grid[max_idx]
        
        return float(u_best), float(v_best)
        
    except Exception:
        return None, None


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

            lat_name = next((n for n in ["lat","latitude"] if n in da.coords), None)
            lon_name = next((n for n in ["lon","longitude"] if n in da.coords), None)

            if lat_name and lon_name:
                lat_idx = np.abs(da[lat_name].values - lat).argmin()
                lon_idx = np.abs(da[lon_name].values - lon).argmin()

                rain_val = float(da.isel({lat_name: lat_idx, lon_name: lon_idx}).values)

                if np.isnan(rain_val):
                    rain_val = None

        except:
            rain_val = None

    # SMART CURRENT
    u_cur, v_cur = get_current_smart(ds_cur, t, lat, lon)

    return {
        "wave": {
            "hs": safe_extract(ds_wave,"hs",t,lat,lon),
            "tp": safe_extract(ds_wave,"t01",t,lat,lon),
            "dir": safe_extract(ds_wave,"dir",t,lat,lon)
        },
        "wind": {
            "u": safe_extract(ds_wave,"uwnd",t,lat,lon),
            "v": safe_extract(ds_wave,"vwnd",t,lat,lon)
        },
        "current": {
            "u": u_cur,
            "v": v_cur
        },
        "rain": {
            "precip": rain_val
        }
    }


# =========================
# MAIN PROCESS
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

        sample_points = generate_3_points_along_route(segment_route)

        samples = []

        for j, (lat, lon) in enumerate(sample_points):
            t = t0 + timedelta(hours=j * 3)

            sample = extract_hourly_weather(ds_wave, ds_cur, ds_rain, t, lat, lon)
            samples.append(sample)

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
