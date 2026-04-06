# =========================
# MODULE 3 + 4 (FINAL STABLE VERSION)
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
# GSMAP (RESOURCE CACHE)
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
# LOAD DATASET (RESOURCE CACHE)
# =========================
@st.cache_resource(ttl=3600)
def load_datasets_cached(dt_input):

    dt = normalize_date(dt_input)
    if dt is None:
        return None, None, None

    user = st.secrets["bmkg"]["user"]
    password = st.secrets["bmkg"]["pass"]

    YYYY, MM, DD = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")

    # WW3
    ds_wave = None
    for url in [
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/ww3gfs/{YYYY}/{MM}/w3g_hires_{YYYY}{MM}{DD}_1200.nc",
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/ww3gfs/{YYYY}/{MM}/w3g_hires_{YYYY}{MM}{DD}_0000.nc",
    ]:
        try:
            ds_wave = xr.open_dataset(url)
            break
        except:
            time.sleep(1)

    # FVCOM
    ds_cur = None
    for url in [
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/fvcom/{YYYY}/{MM}/InaFlows_{YYYY}{MM}{DD}_1200.nc",
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/fvcom/{YYYY}/{MM}/InaFlows_{YYYY}{MM}{DD}_0000.nc",
    ]:
        try:
            ds_cur = xr.open_dataset(url)
            break
        except:
            time.sleep(1)

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

            # ======================
            # HANDLE TIME
            # ======================
            if "time" in da.dims:
                da = da.sel(time=t, method="nearest")

            # ======================
            # DETECT NAMA KOORDINAT
            # ======================
            lat_name = None
            for name in ["lat", "latitude"]:
                if name in da.coords:
                    lat_name = name
                    break

            lon_name = None
            for name in ["lon", "longitude"]:
                if name in da.coords:
                    lon_name = name
                    break

            # ======================
            # AMBIL NILAI TERDEKAT
            # ======================
            if lat_name and lon_name:

                lat_vals = da[lat_name].values
                lon_vals = da[lon_name].values

                lat_idx = np.abs(lat_vals - lat).argmin()
                lon_idx = np.abs(lon_vals - lon).argmin()

                da_point = da.isel({lat_name: lat_idx, lon_name: lon_idx})

                rain_val = float(da_point.values)

                if np.isnan(rain_val):
                    rain_val = None

        except:
            rain_val = None

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
            "u": safe_extract(ds_cur,"u",t,lat,lon,depth=0.5),
            "v": safe_extract(ds_cur,"v",t,lat,lon,depth=0.5)
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

    for i in range(4):

        lat, lon = route[min(i, len(route)-1)]

        t0 = dt_utc0 + timedelta(hours=i * 6)
        t3 = t0 + timedelta(hours=3)

        sample0 = extract_hourly_weather(ds_wave, ds_cur, ds_rain, t0, lat, lon)
        sample3 = extract_hourly_weather(ds_wave, ds_cur, ds_rain, t3, lat, lon)

        segments.append({
            "interval": f"T{i*6}-T{(i+1)*6}",
            "samples": [sample0, sample3]
        })

    return {
        "tanggal": dt_local,
        "tz": tz,
        "segments": segments
    }
