import streamlit as st
import pandas as pd
from pathlib import Path
import datetime
import re

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
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =========================
# 🔥 PARSER DMS → DECIMAL (UPDATED)
# =========================
def dms_single_to_decimal(dms):
    pattern = r"(\d+)[°º]\s*(\d+)'\s*(\d+)\"\s*([NSEW])"
    match = re.search(pattern, dms.strip())

    if not match:
        return None

    deg, minute, sec, direction = match.groups()

    decimal = float(deg) + float(minute)/60 + float(sec)/3600

    if direction in ['S', 'W']:
        decimal *= -1

    return decimal


def parse_coordinate_pair(text):
    parts = text.split("-")

    if len(parts) != 2:
        return None

    lat = dms_single_to_decimal(parts[0])
    lon = dms_single_to_decimal(parts[1])

    if lat is None or lon is None:
        return None

    return f"{lat},{lon}"


def parse_full_coordinate(text):
    try:
        start_text, end_text = text.split("To")

        start = parse_coordinate_pair(start_text.strip())
        end = parse_coordinate_pair(end_text.strip())

        if not start or not end:
            return None

        return {
            "awal": start,
            "akhir": end
        }

    except:
        return None

# =========================
# MODE INPUT
# =========================
st.header("🟦 Mode Input Data")

mode = st.radio(
    "Pilih metode:",
    ["Ambil dari Google Sheet", "Input Manual"]
)

# =========================
# MODE 1 – GOOGLE SHEET (ASLI)
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
# MODE 2 – INPUT MANUAL (UPDATED)
# =========================
else:

    st.header("📝 Input Manual Data Permintaan")

    nama = st.text_input("Nama Perusahaan")
    alamat = st.text_input("Alamat Perusahaan")
    nomor = st.text_input("Nomor Surat")
    tanggal = st.date_input("Tanggal")

    koordinat_dms = st.text_area(
        "Koordinat (format derajat dari surat)",
        placeholder="1º 41' 36\" N-101º 28' 38\" E To 1º 11' 07\" N-103º 50' 06\" E"
    )

    if st.button("Simpan Data Manual") and not st.session_state.manual_saved:

        # VALIDASI FORMAT
        if "-" not in koordinat_dms or "To" not in koordinat_dms:
            st.error("Format harus: LAT-LON To LAT-LON")
            st.stop()

        parsed = parse_full_coordinate(koordinat_dms)

        if parsed is None:
            st.error("Format koordinat tidak valid")
            st.stop()

        id_surat = generate_id()

        data = {
            "Id": id_surat,
            "Requester": "manual",
            "Timestamp": str(datetime.datetime.now()),
            "Nama Perusahaan": nama or "-",
            "Alamat Perusahaan": alamat or "-",
            "Nomor Surat": nomor or "-",
            "Informasi": "-",
            "Tanggal Koordinat": str(tanggal),
            "Koordinat": koordinat_dms,
            "Koordinat Awal": koordinat_dms.split("To")[0],
            "Koordinat Akhir": koordinat_dms.split("To")[1],
            "Koordinat Awal (Desimal)": parsed["awal"],
            "Koordinat Akhir (Desimal)": parsed["akhir"],
            "Water Checker Awal": "",
            "Water Checker Akhir": ""
        }

        save_manual_input(data)

        st.session_state.manual_saved = True

        st.success(f"Data tersimpan dengan ID: {id_surat}")

        df_id = pd.DataFrame([data])
        st.dataframe(df_id)

    if not st.session_state.manual_saved:
        st.stop()

# =========================
# MODULE 2 – ROUTE (ASLI)
# =========================
st.header("🟩 Input Lokasi / Rute")

if "results_module2_dict" not in st.session_state:
    st.session_state.results_module2_dict = {}

if df_id is None or len(df_id) == 0:
    st.warning("Data ID belum tersedia")
    st.stop()

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
# MODULE 3-4 (ASLI)
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
# MODULE 5 (ASLI)
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
# MODULE 6 (ASLI)
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
# DOWNLOAD (ASLI)
# =========================
if st.session_state.doc_buffer:
    st.download_button(
        "⬇️ Download Laporan",
        data=st.session_state.doc_buffer,
        file_name=f"PTSP_{'manual' if mode=='Input Manual' else selected_id}.docx"
    )
