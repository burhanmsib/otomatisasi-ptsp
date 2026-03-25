# =========================
# MODULE 3 + 4 (OPTIMIZED + GSMAP ACTIVE)
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
# CREDENTIAL
# =========================
def get_bmkg_credentials():
    return (
        st.secrets["bmkg"]["user"],
        st.secrets["bmkg"]["pass"]
    )

# =========================
# DATE NORMALIZATION
# =========================
def normalize_date(raw):

    if raw is None or str(raw).strip() == "":
        return None

    s = str(raw)

    # =========================
    # HAPUS JAM (kalau ada)
    # =========================
    s = re.sub(r"\d{1,2}[.:]\d{2}(-\d{1,2}[.:]\d{2})?", "", s)

    # =========================
    # NORMALISASI SEPARATOR
    # =========================
    s = s.replace("/", " ")

    # =========================
    # KONVERSI BULAN INDONESIA → INGGRIS
    # =========================
    month_map = {
        "Januari": "January",
        "Februari": "February",
        "Maret": "March",
        "April": "April",
        "Mei": "May",
        "Juni": "June",
        "Juli": "July",
        "Agustus": "August",
        "September": "September",
        "Oktober": "October",
        "November": "November",
        "Desember": "December"
    }

    for indo, eng in month_map.items():
        s = s.replace(indo, eng)

    s = s.strip()

    # =========================
    # FORMAT YANG DIDUKUNG
    # =========================
    formats = [
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%d %B %Y",
        "%Y-%m-%d",
        "%d %b %Y",
        "%d/%m/%Y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except:
            continue

    # =========================
    # FALLBACK PARSER
    # =========================
    try:
        return parser.parse(s, dayfirst=True)
    except:
        return None

# =========================
# LOAD GSMAP (CACHED)
# =========================
@st.cache_data(ttl=3600)
def load_gsmap_cached(dt):

    ftp_host = st.secrets["ftp"]["host"]
    ftp_user = st.secrets["ftp"]["user"]
    ftp_pass = st.secrets["ftp"]["pass"]

    Y = dt.strftime("%Y")
    M = dt.strftime("%m")
    D = dt.strftime("%d")
    H = dt.strftime("%H")

    remote_path = f"/himawari6/GSMaP/netcdf/{Y}/{M}/{D}/GSMaP_{Y}{M}{D}{H}00.nc"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
    tmp_path = tmp.name
    tmp.close()

    ftp = ftplib.FTP(ftp_host)
    ftp.login(ftp_user, ftp_pass)

    with open(tmp_path, "wb") as f:
        ftp.retrbinary(f"RETR {remote_path}", f.write)

    ftp.quit()

    ds = xr.open_dataset(tmp_path)

    os.remove(tmp_path)

    return ds

# =========================
# LOAD DATASET (CACHED)
# =========================
@st.cache_data(ttl=3600)
def load_datasets_cached(dt_utc):

    user, password = get_bmkg_credentials()

    YYYY, MM, DD = dt_utc.strftime("%Y"), dt_utc.strftime("%m"), dt_utc.strftime("%d")

    # WW3
    urls_wave = [
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/ww3gfs/{YYYY}/{MM}/w3g_hires_{YYYY}{MM}{DD}_1200.nc",
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/ww3gfs/{YYYY}/{MM}/w3g_hires_{YYYY}{MM}{DD}_0000.nc",
    ]

    ds_wave = None
    for url in urls_wave:
        try:
            ds_wave = xr.open_dataset(url)
            break
        except:
            time.sleep(1)

    if ds_wave is None:
        return None, None, None

    # FVCOM
    urls_cur = [
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/fvcom/{YYYY}/{MM}/InaFlows_{YYYY}{MM}{DD}_1200.nc",
        f"https://{user}:{password}@maritim.bmkg.go.id/opendap/fvcom/{YYYY}/{MM}/InaFlows_{YYYY}{MM}{DD}_0000.nc",
    ]

    ds_cur = None
    for url in urls_cur:
        try:
            ds_cur = xr.open_dataset(url)
            break
        except:
            time.sleep(1)

    if ds_cur is None:
        return None, None, None

    # GSMAP → hanya load 1x (jam 00)
    ds_rain = load_gsmap_cached(dt_utc)

    return ds_wave, ds_cur, ds_rain

# =========================
# SAFE EXTRACT (FAST)
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

        try:
            val = float(da.sel(lat=lat, lon=lon, method="nearest").values)
            if not np.isnan(val):
                return val
        except:
            pass

        lat_vals = ds["lat"].values
        lon_vals = ds["lon"].values

        lat_idx = np.abs(lat_vals - lat).argmin()
        lon_idx = np.abs(lon_vals - lon).argmin()

        for r in range(1, 3):
            for i in range(lat_idx-r, lat_idx+r+1):
                for j in range(lon_idx-r, lon_idx+r+1):
                    try:
                        val = float(da.isel(lat=i, lon=j).values)
                        if not np.isnan(val):
                            return val
                    except:
                        continue

        return 0.0

    except:
        return 0.0

# =========================
# EXTRACT WEATHER
# =========================
def extract_hourly_weather(ds_wave, ds_cur, ds_rain, t, lat, lon):

    rain_val = None

    try:
        var = list(ds_rain.data_vars)[0]
        da = ds_rain[var]

        if "time" in da.dims:
            da = da.sel(time=t, method="nearest")

        rain_val = float(da.sel(lat=lat, lon=lon, method="nearest").values)

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
