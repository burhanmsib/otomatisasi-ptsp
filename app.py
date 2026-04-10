import streamlit as st
import pandas as pd
from pathlib import Path
import datetime
import re
import pytz

# =========================
# IMPORT MODULE
# =========================
from modules.module1_request import (
    load_request_sheet_streamlit,
    save_manual_input,
    generate_id
)
from modules.module2_route import process_route_segment_module2_streamlit
from modules.module34_data import process_module34, load_datasets_cached
from modules.module5_analysis import process_module5
from modules.module6_report import generate_final_docx_streamlit

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="PTSP Marine Meteorological Report",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 PTSP Marine Meteorological Report Automation")

# =========================
# INIT SESSION STATE
# =========================
def init_state():
    keys = {
        "df_requests": None,
        "selected_id": None,
        "results_module2": None,
        "results_module34": None,
        "results_module5": None,
        "doc_buffer": None,
        "run_module34": False,
        "run_module5": False,
        "run_generate": False,
        "ds_wave": None,
        "ds_cur": None,
        "ds_rain": None,
        "manual_saved": False,
        "preview_data": None,
        "df_id_manual": None,   # 🔥 TAMBAHAN FIX
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =========================
# FIX df_id
# =========================
df_id = None

# =========================
# PARSER FLEXIBLE
# =========================

def dms_to_decimal(match):
    deg = float(match.group(1))
    minute = match.group(2)
    sec = match.group(3)
    direction = match.group(4)

    minute = float(minute) if minute else 0
    sec = float(sec) if sec else 0

    # 🔥 DETEKSI DMM vs DMS
    if sec == 0 and minute != 0:
        # 👉 kemungkinan besar DMM
        decimal = deg + (minute / 60)
    else:
        # 👉 DMS normal
        decimal = deg + (minute / 60) + (sec / 3600)

    if direction in ['S', 'W']:
        decimal *= -1

    return decimal

import re

def normalize_text(text):
    text = text.upper()

    # normalisasi simbol
    text = text.replace("̊", "°")
    text = text.replace("º", "°")
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("”", '"')
    text = text.replace("“", '"')

    # hilangkan kata pengganggu
    text = text.replace("FROM", "")
    text = text.replace("TO", "")

    return text


def dms_to_decimal(deg, minute=0, second=0, direction="N"):
    val = float(deg) + float(minute)/60 + float(second)/3600
    if direction in ["S", "W"]:
        val *= -1
    return val


def extract_coordinates(text):
    text = normalize_text(text)

    results = []

    # =========================
    # 1. FORMAT DECIMAL
    # =========================
    decimal_matches = re.findall(r'(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)', text)
    for lat, lon in decimal_matches:
        results.append((float(lat), float(lon)))

    # =========================
    # 2. FORMAT DMS / DMM / CAMPUR
    # =========================
    pattern = re.findall(
        r'(\d{1,3})[°\s-]*(\d{1,2}(?:\.\d+)?)?\'?\s*(\d{1,2}(?:\.\d+)?)?"?\s*([NS])'
        r'.{0,10}?'
        r'(\d{1,3})[°\s-]*(\d{1,2}(?:\.\d+)?)?\'?\s*(\d{1,2}(?:\.\d+)?)?"?\s*([EW])',
        text
    )

    for m in pattern:
        lat_deg, lat_min, lat_sec, lat_dir, lon_deg, lon_min, lon_sec, lon_dir = m

        lat = dms_to_decimal(
            lat_deg,
            lat_min or 0,
            lat_sec or 0,
            lat_dir
        )

        lon = dms_to_decimal(
            lon_deg,
            lon_min or 0,
            lon_sec or 0,
            lon_dir
        )

        results.append((lat, lon))

    return results


def parse_coordinate(text):
    coords = extract_coordinates(text)

    if not coords:
        return None

    # minimal 1 titik
    if len(coords) == 1:
        return {
            "awal": f"{coords[0][0]},{coords[0][1]}",
            "akhir": f"{coords[0][0]},{coords[0][1]}",
            "all": coords
        }

    return {
        "awal": f"{coords[0][0]},{coords[0][1]}",
        "akhir": f"{coords[-1][0]},{coords[-1][1]}",
        "all": coords
    }

# =========================
# MODE INPUT
# =========================
st.header("🟦 Mode Input Data")

mode = st.radio(
    "Pilih metode:",
    ["Ambil dari Google Sheet", "Input Manual"]
)

if mode == "Ambil dari Google Sheet":
    st.session_state.manual_saved = False

# =========================
# MODE 1 – GOOGLE SHEET
# =========================
if mode == "Ambil dari Google Sheet":

    st.header("🟦 Data Permintaan PTSP")

    df_requests = load_request_sheet_streamlit()

    if df_requests is None:
        st.error("Gagal load data")
        st.stop()

    st.session_state.df_requests = df_requests

    st.header("🆔 Pilih ID Surat")

    id_list = sorted(df_requests["Id"].astype(str).unique())

    col1, col2 = st.columns(2)

    with col1:
        selected_id_dropdown = st.selectbox("Pilih dari daftar", [""] + id_list)

    with col2:
        selected_id_manual = st.text_input("Atau input ID manual")

    selected_id = selected_id_manual if selected_id_manual else selected_id_dropdown

    if not selected_id:
        st.warning("Silakan pilih atau input ID terlebih dahulu")
        st.stop()

    df_id = df_requests[df_requests["Id"].astype(str) == selected_id]

    if df_id.empty:
        st.error("Data tidak ditemukan")
        st.stop()

    st.success(f"{len(df_id)} data ditemukan")
    st.dataframe(df_id)

# =========================
# MODE 2 – INPUT MANUAL
# =========================
else:

    st.header("📝 Input Manual Data Permintaan")

    requester = st.text_input("Nama FOD")
    nama = st.text_input("Nama Perusahaan")
    alamat = st.text_input("Alamat Perusahaan")
    nomor = st.text_input("Nomor Surat")

    jumlah = st.number_input("Jumlah Titik Permintaan", min_value=1, step=1)

    data_list = []

    for i in range(jumlah):
        st.subheader(f"Titik {i+1}")

        tanggal_i = st.date_input(f"Tanggal {i+1}", key=f"tgl_{i}")
        koordinat_i = st.text_area(f"Koordinat {i+1}", key=f"coord_{i}")

        data_list.append({
            "tanggal": tanggal_i,
            "koordinat": koordinat_i
        })

    # =========================
    # PREVIEW
    # =========================
    if st.button("Preview Data"):

        parsed_rows = []

        for d in data_list:
            parsed = parse_coordinate(d["koordinat"])

            if parsed is None:
                st.error(f"Koordinat tidak valid: {d['koordinat']}")
                st.stop()

            parsed_rows.append({
                "Tanggal Koordinat": str(d["tanggal"]),
                "Koordinat": d["koordinat"],
                "Koordinat Awal (Desimal)": parsed["awal"],
                "Koordinat Akhir (Desimal)": parsed["akhir"]
            })

        df_preview = pd.DataFrame(parsed_rows)
        st.session_state.preview_data = df_preview

        st.dataframe(df_preview)

    # =========================
    # SAVE
    # =========================
    if st.session_state.preview_data is not None:

        if st.button("Simpan Data Manual"):

            jakarta_tz = pytz.timezone("Asia/Jakarta")
            now_wib = datetime.datetime.now(jakarta_tz)

            id_surat = generate_id()

            for _, row in st.session_state.preview_data.iterrows():

                data = {
                    "Id": id_surat,
                    "Requester": requester or "unknown",
                    "Timestamp": now_wib.strftime("%Y-%m-%d %H:%M:%S"),
                    "Nama Perusahaan": nama or "-",
                    "Alamat Perusahaan": alamat or "-",
                    "Nomor Surat": nomor or "-",
                    "Informasi": "-",
                    "Tanggal Koordinat": row["Tanggal Koordinat"],
                    "Koordinat": row["Koordinat"],
                    "Koordinat Awal": row["Koordinat"],
                    "Koordinat Akhir": row["Koordinat"],
                    "Koordinat Awal (Desimal)": row["Koordinat Awal (Desimal)"],
                    "Koordinat Akhir (Desimal)": row["Koordinat Akhir (Desimal)"],
                    "Water Checker Awal": "",
                    "Water Checker Akhir": ""
                }

                save_manual_input(data)

            st.success(f"Data tersimpan dengan ID: {id_surat}")
            st.code(id_surat)

            # 🔥 FIX UTAMA (JANGAN DIHAPUS)
            df_id = pd.DataFrame([
                {
                    "Id": id_surat,
                    "Tanggal Koordinat": row["Tanggal Koordinat"],
                    "Koordinat": row["Koordinat"],
                    "Koordinat Awal": row["Koordinat"],
                    "Koordinat Akhir": row["Koordinat"],
                    "Koordinat Awal (Desimal)": row["Koordinat Awal (Desimal)"],
                    "Koordinat Akhir (Desimal)": row["Koordinat Akhir (Desimal)"],
                }
                for _, row in st.session_state.preview_data.iterrows()
            ])

            # 🔥 SIMPAN KE SESSION
            st.session_state.df_id_manual = df_id
            st.session_state.manual_saved = True

    if not st.session_state.manual_saved:
        st.stop()

# =========================
# 🔥 FIX STATE df_id (WAJIB)
# =========================
if df_id is None and st.session_state.df_id_manual is not None:
    df_id = st.session_state.df_id_manual

# =========================
# VALIDASI df_id
# =========================
if df_id is None or df_id.empty:
    st.warning("Data belum siap")
    st.stop()

# =========================
# MODULE 2
# =========================
st.header("🟩 Input Lokasi / Rute")

if "results_module2_dict" not in st.session_state:
    st.session_state.results_module2_dict = {}

index_list = list(range(len(df_id)))

selected_index = st.selectbox(
    "Pilih titik yang ingin diinput",
    index_list,
    format_func=lambda x: f"Titik {x+1} - {df_id.iloc[x]['Tanggal Koordinat']}"
)

row = df_id.iloc[selected_index]

hasil = process_route_segment_module2_streamlit(row, selected_index)

if hasil is not None:
    st.session_state.results_module2_dict[selected_index] = hasil
    st.success(f"Titik {selected_index+1} tersimpan")

if len(st.session_state.results_module2_dict) == len(df_id):

    st.session_state.results_module2 = [
        st.session_state.results_module2_dict[i]
        for i in range(len(df_id))
    ]

    st.success("✅ Semua titik/rute sudah dibuat")

# =========================
# MODULE 3-4
# =========================
st.header("🟨 Ambil Data Cuaca")

tz = st.selectbox("Zona Waktu", ["WIB", "WITA", "WIT"])

if "results_module2_dict" not in st.session_state or len(st.session_state.results_module2_dict) == 0:
    st.warning("Silakan isi minimal 1 titik terlebih dahulu")
    st.stop()

if len(st.session_state.results_module2_dict) != len(df_id):
    st.warning("Semua titik harus diisi sebelum lanjut")
    st.stop()

if st.button("🌐 Ambil Data Cuaca"):
    st.session_state.run_module34 = True

if st.session_state.get("run_module34", False):

    if st.session_state.get("ds_wave") is None:

        with st.spinner("Load dataset (sekali saja)..."):

            sample_row = df_id.iloc[0]
            dt_sample = sample_row["Tanggal Koordinat"]

            ds_wave, ds_cur, ds_rain = load_datasets_cached(dt_sample)

            if ds_wave is None or ds_cur is None:
                st.error("Gagal load dataset BMKG")
                st.stop()

            st.session_state.ds_wave = ds_wave
            st.session_state.ds_cur = ds_cur
            st.session_state.ds_rain = ds_rain

    results_module34 = []
    gagal = False

    progress = st.progress(0)

    keys = sorted(st.session_state.results_module2_dict.keys())

    with st.spinner("Mengambil data cuaca..."):

        for idx, i in enumerate(keys):

            progress.progress((idx + 1) / len(keys))

            item = st.session_state.results_module2_dict[i]

            if i >= len(df_id):
                st.error(f"Index {i} melebihi jumlah data")
                gagal = True
                break

            row = df_id.iloc[i]

            result = process_module34(
                row=row,
                polyline=item["titik5"],
                tz=tz,
                ds_wave=st.session_state.ds_wave,
                ds_cur=st.session_state.ds_cur,
                ds_rain=st.session_state.ds_rain
            )

            if result is None:
                gagal = True
                break

            results_module34.append(result)

    if gagal:
        st.error("❌ Gagal mengambil data cuaca")
        st.session_state.results_module34 = None
    else:
        st.success("✅ Data cuaca berhasil")
        st.session_state.results_module34 = results_module34

    st.session_state.run_module34 = False

# =========================
# MODULE 5
# =========================
st.header("🟧 Analisis Cuaca")

if st.button("📊 Jalankan Analisis"):
    st.session_state.run_module5 = True

if st.session_state.run_module5 and st.session_state.results_module34:

    with st.spinner("Analisis..."):

        results_module5 = process_module5(
            st.session_state.results_module34,
            tz=tz
        )

    st.session_state.results_module5 = results_module5
    st.success("✅ Analisis selesai")

    st.session_state.run_module5 = False

# =========================
# MODULE 6
# =========================
st.header("🟥 Generate Laporan")

template_path = Path("templates/Template PTSP.docx")

if not template_path.exists():
    st.error("Template tidak ditemukan")
    st.stop()

if st.button("📄 Generate Laporan"):
    st.session_state.run_generate = True

if st.session_state.run_generate and st.session_state.results_module5:

    with st.spinner("Menyusun laporan..."):

        doc_buffer = generate_final_docx_streamlit(
            module1_rows=df_id.to_dict(orient="records"),
            module5_rows=st.session_state.results_module5,
            template_path=str(template_path)
        )

    st.session_state.doc_buffer = doc_buffer
    st.success("✅ Laporan berhasil dibuat")

    st.session_state.run_generate = False

# =========================
# DOWNLOAD
# =========================
if st.session_state.doc_buffer:
    st.download_button(
        "⬇️ Download Laporan",
        data=st.session_state.doc_buffer,
        file_name=f"PTSP_{'manual' if mode=='Input Manual' else selected_id}.docx"
    )
