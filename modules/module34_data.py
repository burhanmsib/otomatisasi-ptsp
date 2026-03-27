# MODULE 3 + 4 (FIXED - NO SYNTAX ERROR)

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

TZ_OFFSET = {
    "WIB": 7,
    "WITA": 8,
    "WIT": 9
}

# =========================
# DATE
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
# GSMAP
# =========================
@st.cache_resource(ttl=3600)
def load_gsmap_cached(dt):

    try:
        ftp = ftplib.FTP(st.secrets["ftp"]["host"])
        ftp.login(st.secrets["ftp"]["user"], st.secrets["ftp"]["pass"])

        Y, M, D, H = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"), dt.strftime("%H")

        path = f"/himawari6/GSMaP/netcdf/{Y}/{M}/{D}/GSMaP_{Y}{M}{D}{H}00.nc"

        tmp = tempfile.NamedTemporaryFile(delete=False)
        ftp.retrbinary(f"RETR {path}", tmp.write)
        ftp.quit()

        ds = xr.open_dataset(tmp.name)
        os.remove(tmp.name)

        return ds

    except:
        return None

# =========================
# LOAD DATA
# =========================
@st.cache_resource(ttl=3600)
def load_datasets_cached(dt_input):

    dt = normalize_date(dt_input)
    if dt is None:
        return None, None, None

    user = st.secrets["bmkg"]["user"]
    password = st.secrets["bmkg"]["pass"]

    Y, M, D = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")

    ds_wave = None
    try:
        url = f"https://{user}:{password}@maritim.bmkg.go.id/opendap/ww3gfs/{Y}/{M}/w3g_hires_{Y}{M}{D}_1200.nc"
        ds_wave = xr.open_dataset(url)
    except:
        pass

    ds_cur = None
    try:
        url = f"https://{user}:{password}@maritim.bmkg.go.id/opendap/fvcom/{Y}/{M}/InaFlows_{Y}{M}{D}_1200.nc"
        ds_cur = xr.open_dataset(url)
    except:
        pass

    ds_rain = load_gsmap_cached(dt)

    return ds_wave, ds_cur, ds_rain

# =========================
# SAFE EXTRACT
# =========================
def safe_extract(ds, var, t, lat, lon):

    try:
        da = ds[var]
        if "time" in da.dims:
            da = da.sel(time=t, method="nearest")

        return float(da.sel(lat=lat, lon=lon, method="nearest").values)
    except:
        return 0.0

# =========================
# CACHE GRID FVCOM
# =========================
@st.cache_resource
def prepare_fvcom_grid(ds):

    lat_name = "latc" if "latc" in ds else "lat"
    lon_name = "lonc" if "lonc" in ds else "lon"

    return ds[lat_name].values.flatten(), ds[lon_name].values.flatten()

# =========================
# CURRENT (FAST)
# =========================
def safe_extract_current(ds, var, t, lat, lon):

    try:
        da = ds[var]

        if "time" in da.dims:
            da = da.sel(time=t, method="nearest")

        lat_vals, lon_vals = prepare_fvcom_grid(ds)
        data_vals = da.values.flatten()

        dist = (lat_vals - lat)**2 + (lon_vals - lon)**2

        idxs = np.argpartition(dist, 20)[:20]

        for i in idxs:
            val = data_vals[i]
            if not np.isnan(val):
                return float(val)

        return 0.0

    except:
        return 0.0

# =========================
# WEATHER
# =========================
def extract_hourly_weather(ds_wave, ds_cur, ds_rain, t, lat, lon):

    rain_val = None

    if ds_rain is not None:
        try:
            var = list(ds_rain.data_vars)[0]
            da = ds_rain[var]

            if "time" in da.dims:
                da = da.sel(time=t, method="nearest")

            rain_val = float(da.values.flatten()[0])
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
            "u": safe_extract_current(ds_cur,"u",t,lat,lon),
            "v": safe_extract_current(ds_cur,"v",t,lat,lon)
        },
        "rain": {
            "precip": rain_val
        }
    }

# =========================
# PROCESS
# =========================
def process_module34(row, polyline, tz="WIB", ds_wave=None, ds_cur=None, ds_rain=None):

    dt_local = normalize_date(row["Tanggal Koordinat"])
    if dt_local is None:
        return None

    dt_utc0 = dt_local - timedelta(hours=7)

    route = polyline if len(polyline) > 1 else [polyline[0]]*5

    segments = []

    for i in range(4):

        lat, lon = route[min(i, len(route)-1)]

        t0 = dt_utc0 + timedelta(hours=i*6)
        t3 = t0 + timedelta(hours=3)

        segments.append({
            "interval": f"T{i*6}-T{(i+1)*6}",
            "samples": [
                extract_hourly_weather(ds_wave, ds_cur, ds_rain, t0, lat, lon),
                extract_hourly_weather(ds_wave, ds_cur, ds_rain, t3, lat, lon)
            ]
        })

    return {
        "tanggal": dt_local,
        "segments": segments
    }
