import streamlit as st
import pandas as pd
from pathlib import Path

# =========================
# IMPORT MODULE
# =========================
from modules.module1_request import load_request_sheet_streamlit
from modules.module2_route import process_route_segment_module2_streamlit
from modules.module34_data import process_module34
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
if df_requests is not None:
    st.session_state.df_requests = df_requests
else:
    st.warning("Data tidak tersedia")
    st.stop()

# =========================
# PILIH ID
# =========================
st.header("🆔 Pilih ID Surat")

id_list = sorted(df_requests["Id"].astype(str).unique())

selected_id = st.selectbox("Pilih ID", id_list)

if selected_id:
    st.session_state.selected_id = selected_id

# =========================
# FILTER DATA
# =========================
df_id = df_requests[df_requests["Id"].astype(str) == selected_id]

if df_id.empty:
    st.warning("Data tidak ditemukan")
else:
    st.success(f"{len(df_id)} data ditemukan")
    st.dataframe(df_id)

# =========================
# MODULE 2 – ROUTE
# =========================
st.header("🟩 Gambar Rute")

results_module2 = []

for idx, row in df_id.iterrows():

    st.markdown(f"### 📍 {row['Tanggal Koordinat']}")

    hasil = process_route_segment_module2_streamlit(row, idx)

    if hasil is not None:
        results_module2.append(hasil)

if len(results_module2) == len(df_id):
    st.session_state.results_module2 = results_module2
    st.success("✅ Semua rute valid")

# =========================
# MODULE 3-4 BUTTON
# =========================
st.header("🟨 Ambil Data Cuaca")

tz = st.selectbox("Zona Waktu", ["WIB", "WITA", "WIT"])

if st.button("🌐 Ambil Data Cuaca"):
    st.session_state.run_module34 = True

# =========================
# MODULE 3-4 PROCESS
# =========================
if st.session_state.run_module34 and st.session_state.results_module2:

    results_module34 = []
    gagal = False

    with st.spinner("Mengambil data cuaca..."):

        for i, item in enumerate(st.session_state.results_module2):

            result = process_module34(
                row=df_id.iloc[i],
                polyline=item["titik5"],
                tz=tz
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
# MODULE 5 BUTTON
# =========================
st.header("🟧 Analisis Cuaca")

if st.button("📊 Jalankan Analisis"):
    st.session_state.run_module5 = True

# =========================
# MODULE 5 PROCESS
# =========================
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
# MODULE 6 BUTTON
# =========================
st.header("🟥 Generate Laporan")

template_path = Path("templates/Template PTSP.docx")

if st.button("📄 Generate Laporan"):
    st.session_state.run_generate = True

# =========================
# MODULE 6 PROCESS
# =========================
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
# DEBUG (optional)
# =========================
with st.expander("DEBUG STATE"):
    st.write(st.session_state)
