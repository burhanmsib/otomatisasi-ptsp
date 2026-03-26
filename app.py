import streamlit as st
import pandas as pd
from pathlib import Path

# =========================
# IMPORT MODULE
# =========================
from modules.module1_request import load_request_sheet_streamlit
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
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =========================
# MODULE 1 – GOOGLE SHEET
# =========================
st.header("🟦 Data Permintaan PTSP")

df_requests = load_request_sheet_streamlit()

if df_requests is None:
    st.error("Gagal load data")
    st.stop()

st.session_state.df_requests = df_requests

# =========================
# PILIH ID
# =========================
st.header("🆔 Pilih ID Surat")

id_list = sorted(df_requests["Id"].astype(str).unique())

st.subheader("Pilih atau Input ID")

col1, col2 = st.columns(2)

with col1:
    selected_id_dropdown = st.selectbox("Pilih dari daftar", [""] + id_list)

with col2:
    selected_id_manual = st.text_input("Atau input ID manual")

selected_id = selected_id_manual if selected_id_manual else selected_id_dropdown

if not selected_id:
    st.warning("Silakan pilih atau input ID")
    st.stop()
# =========================
# MODULE 2 – ROUTE
# =========================
st.header("🟩 Input Lokasi / Rute")

# =========================
# INIT STATE
# =========================
if "results_module2_dict" not in st.session_state:
    st.session_state.results_module2_dict = {}

# =========================
# PILIH TITIK
# =========================
index_list = list(range(len(df_id)))

selected_index = st.selectbox(
    "Pilih titik yang ingin diinput",
    index_list,
    format_func=lambda x: f"Titik {x+1} - {df_id.iloc[x]['Tanggal Koordinat']}"
)

row = df_id.iloc[selected_index]

# =========================
# PROSES MODULE 2
# =========================
hasil = process_route_segment_module2_streamlit(row, selected_index)

if hasil is not None:
    st.session_state.results_module2_dict[selected_index] = hasil
    st.success(f"Titik {selected_index+1} tersimpan")

# =========================
# CEK SEMUA SELESAI
# =========================
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

if st.button("🌐 Ambil Data Cuaca"):
    st.session_state.run_module34 = True

# =========================
# PROCESS MODULE 3-4
# =========================
if st.session_state.run_module34 and st.session_state.results_module2:

    # =========================
    # LOAD DATASET SEKALI
    # =========================
    if st.session_state.ds_wave is None:

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

    # =========================
    # PROCESS LOOP
    # =========================
    results_module34 = []
    gagal = False

    progress = st.progress(0)

    with st.spinner("Mengambil data cuaca..."):

        for i, item in enumerate(st.session_state.results_module2):

            progress.progress((i + 1) / len(st.session_state.results_module2))

            result = process_module34(
                row=df_id.iloc[i],
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
        st.error("❌ Gagal ambil data cuaca")
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
        file_name=f"PTSP_{selected_id}.docx"
    )

# =========================
# DEBUG
# =========================
with st.expander("DEBUG STATE"):
    st.write(st.session_state)
