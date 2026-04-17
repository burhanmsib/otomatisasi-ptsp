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

import re

# =========================
# NORMALIZE TEXT (SUPER FLEX)
# =========================
def normalize_text(text):
    text = text.upper()

    # simbol derajat
    text = text.replace("O", "°")
    text = text.replace("º", "°")
    text = text.replace("˚", "°")
    text = text.replace("̊", "°")

    # petik
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("”", '"').replace("“", '"')

    # ubah DMM (42'068 → 42.068)
    # hanya convert kalau TIDAK ADA titik (biar tidak merusak DMM)
    text = re.sub(r"(\d+)'(\d{3})\b", r"\1.\2", text)

    # pisahin angka & arah
    text = re.sub(r"(\d)([NS])", r"\1 \2", text)
    text = re.sub(r"(\d)([EW])", r"\1 \2", text)
    text = re.sub(r"([NSEW])(\d)", r"\1 \2", text)

    # separator
    text = text.replace(" TO ", " | ")
    text = text.replace("–", " | ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    # text = re.sub(r'\s-\s', ' ', text)
    # text = re.sub(r'([NS])-(\d)', r'\1 -\2', text)
    
    # ubah double petik aneh
    text = text.replace("’’", '"')
    text = text.replace("''", '"')
    text = text.replace('""', '"')

    # JANGAN hapus koma → penting untuk decimal
    # text = text.replace(",", " ")

    # bersihin kata
    text = text.replace("FROM", "")
    text = text.replace("KE", "")
    text = text.replace("DARI", "")

    # rapihin spasi
    text = " ".join(text.split())
    text = text.replace("S-", "S -")
    text = text.replace("N-", "N -")

    # pisahin angka besar (lebih aman)
    text = re.sub(r"(\d{3})(\d{2})", r"\1 \2", text)

    # jaga-jaga separator
    text = text.replace("||", "|")

    return text


# =========================
# DMS → DECIMAL (SMART)
# =========================
def dms_to_decimal(deg, minute=0, second=0, direction="N"):
    deg = float(deg)
    minute = float(minute) if minute else 0
    second = float(second) if second else 0

    # deteksi DMM vs DMS
    if second == 0 and '.' in str(minute):
        decimal = deg + (minute / 60)
    else:
        decimal = deg + (minute / 60) + (second / 3600)

    if direction in ["S", "W"]:
        decimal *= -1

    return decimal


# =========================
# EXTRACT ALL FORMAT
# =========================
def extract_coordinates(text):
    text = normalize_text(text)
    results = []

    # =========================
    # 1. DECIMAL FORMAT
    # =========================
    decimal_matches = re.findall(r'(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)', text)
    for lat, lon in decimal_matches:
        results.append((float(lat), float(lon)))

    # =========================
    # 2. DMS / DMM FLEX
    # =========================
    pattern = re.findall(
        r'(\d{1,3})\s*°?\s*'
        r'(\d{1,2}(?:\.\d+)?)?\'?\s*'
        r'(\d{1,2}(?:\.\d+)?)?"?\s*'
        r'([NS])'
        r'.{0,15}?'
        r'(\d{1,3})\s*°?\s*'
        r'(\d{1,2}(?:\.\d+)?)?\'?\s*'
        r'(\d{1,2}(?:\.\d+)?)?"?\s*'
        r'([EW])',
        text
    )

    for m in pattern:
        lat_deg, lat_min, lat_sec, lat_dir, lon_deg, lon_min, lon_sec, lon_dir = m

        lat = dms_to_decimal(lat_deg, lat_min or 0, lat_sec or 0, lat_dir)
        lon = dms_to_decimal(lon_deg, lon_min or 0, lon_sec or 0, lon_dir)

        results.append((lat, lon))

    # =========================
    # 3. FALLBACK FORMAT
    # =========================
    fallback = re.findall(
        r'(\d{1,3})\s+(\d{1,2})\s+(\d{1,2})\s*([NS])\s+'
        r'(\d{1,3})\s+(\d{1,2})\s+(\d{1,2})\s*([EW])',
        text
    )

    for m in fallback:
        lat = dms_to_decimal(m[0], m[1], m[2], m[3])
        lon = dms_to_decimal(m[4], m[5], m[6], m[7])
        results.append((lat, lon))

    # =========================
    # 4. FORMAT DMM TANPA °
    # contoh: 06’05.584’’S
    # =========================
    pattern_dmm_no_deg = re.findall(
        r'(\d{1,3})[\'’]\s*(\d{1,2}\.\d+)[\'’’"]*\s*([NS])'
        r'.{0,10}?'
        r'(\d{1,3})[\'’]\s*(\d{1,2}\.\d+)[\'’’"]*\s*([EW])',
        text
    )
    
    for m in pattern_dmm_no_deg:
        lat_deg, lat_min, lat_dir, lon_deg, lon_min, lon_dir = m
    
        lat = dms_to_decimal(lat_deg, lat_min, 0, lat_dir)
        lon = dms_to_decimal(lon_deg, lon_min, 0, lon_dir)
    
        results.append((lat, lon))

    # =========================
    # VALIDASI KOORDINAT
    # =========================
    clean_results = []

    for lat, lon in results:
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            clean_results.append((lat, lon))

    # debug kalau kosong
    if len(clean_results) == 0:
        print("⚠️ Koordinat tidak terbaca:", text)

    return clean_results


def clean_coordinate(text):
    import re

    if not text:
        return text

    # =========================
    # FIX DETIK > 59 (SAMAKAN DENGAN SISTEM ATAS)
    # =========================
    def fix_seconds(match):
        val = match.group(1)
        try:
            num = float(val)

            # 🔥 LOGIKA PENTING (JANGAN DIUBAH)
            if num > 99:
                num = num / 10   # 350 → 35.0
            elif num > 59:
                num = num / 10   # 90 → 9.0

            return f'{num}"'
        except:
            return val + '"'

    text = re.sub(r'(\d+)"', fix_seconds, text)

    # =========================
    # NORMALISASI SPASI
    # =========================
    text = text.replace("  ", " ")
    text = text.replace(" -", " - ")
    text = text.replace("- ", " - ")

    return text.strip()


# =========================
# MAIN PARSER (FINAL FIX)
# =========================
def parse_coordinate(text):

    # 🔥 TAMBAHAN (TIDAK MENGUBAH LOGIC)
    text = clean_coordinate(text)

    coords = extract_coordinates(text)

    if not coords:
        return None

    # 🔵 MODE TITIK (TETAP)
    if len(coords) == 1:
        lat, lon = coords[0]

        return {
            "mode": "titik",
            "Koordinat Awal (Desimal)": f"{lat},{lon}",
            "Koordinat Akhir (Desimal)": f"{lat},{lon}",
            "All Points": coords,

            # 🔥 TAMBAHAN (AGAR CONSISTENT DENGAN CODE LAIN)
            "awal": f"{lat},{lon}",
            "akhir": f"{lat},{lon}",
            "all": coords
        }

    # 🟢 MODE RUTE (TETAP)
    return {
        "mode": "rute",
        "Koordinat Awal (Desimal)": f"{coords[0][0]},{coords[0][1]}",
        "Koordinat Akhir (Desimal)": f"{coords[-1][0]},{coords[-1][1]}",
        "All Points": coords,

        # 🔥 TAMBAHAN (BIAR COMPATIBLE DENGAN APP.PY BARU)
        "awal": f"{coords[0][0]},{coords[0][1]}",
        "akhir": f"{coords[-1][0]},{coords[-1][1]}",
        "all": coords
    }

st.info("ℹ️ Nilai koordinat hasil parsing mungkin berbeda format, namun perhitungan rute dan analisis tetap menggunakan data yang benar.")

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
# MODE 1 – GOOGLE SHEET (FINAL FIX)
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
    # 🔥 PARSE OTOMATIS (FINAL FIX)
    # =========================
    parsed_rows = []
    
    # =========================
    # 🔥 FUNCTION AMBIL DATA FLEXIBLE
    # =========================
    def get_val(row, keys):
        for k in keys:
            if k in row and str(row.get(k)).strip():
                return row.get(k)
        return ""
    
    for _, row in df_id.iterrows():
    
        koordinat = row.get("Koordinat", "")
    
        parsed = parse_coordinate(koordinat)
    
        if parsed is None:
            st.error(f"Koordinat tidak valid: {koordinat}")
            st.stop()
    
        # =========================
        # 🔥 AMBIL DATA (ANTI SALAH KOLOM)
        # =========================
        nama_perusahaan = get_val(row, [
            "Nama Perusahaan", "Perusahaan", "Company", "Client"
        ])
    
        alamat_perusahaan = get_val(row, [
            "Alamat Perusahaan", "Alamat", "Address"
        ])
    
        nomor_surat = get_val(row, [
            "Nomor Surat", "No Surat", "Ref", "Reference"
        ])
    
        parsed_rows.append({
            "Tanggal Koordinat": str(
                row.get("Tanggal Koordinat") 
                or row.get("Tanggal") 
                or row.get("Date") 
                or ""
            ),
    
            "Koordinat": koordinat,
    
            # =========================
            # 🔥 FIX UTAMA (PASTI MASUK)
            # =========================
            "Nama Perusahaan": nama_perusahaan,
            "Alamat Perusahaan": alamat_perusahaan,
            "Nomor Surat": nomor_surat,
    
            # =========================
            # 🔥 KOORDINAT UNTUK REPORT
            # =========================
            "Koordinat Awal": koordinat,
            "Koordinat Akhir": koordinat,
    
            # =========================
            # DESIMAL UNTUK PROCESSING
            # =========================
            "Koordinat Awal (Desimal)": parsed.get("awal"),
            "Koordinat Akhir (Desimal)": parsed.get("akhir"),
    
            "Mode": parsed.get("mode"),
            "All Points": parsed.get("all", [])
        })
    
    df_parsed = pd.DataFrame(parsed_rows)
    # =========================
    # 🔥 SIMPAN KE STATE (INI KUNCI FIX)
    # =========================
    st.session_state.selected_data = df_parsed
    st.session_state.preview_data = df_parsed  # optional (biar tetap bisa preview)

    st.success("✅ Data siap digunakan (tanpa perlu preview)")
    st.dataframe(df_parsed)

    # =========================
    # (OPSIONAL) BUTTON PREVIEW
    # =========================
    if st.button("Preview Data"):
        st.info("Preview sama dengan data di atas")

# =========================
# MODE 2 – INPUT MANUAL
# =========================
else:

    st.header("📝 Input Manual Data Permintaan")

    requester = st.text_input("Nama FOD")
    nama = st.text_input("Nama Perusahaan")
    alamat = st.text_input("Alamat Perusahaan")
    nomor = st.text_input("Nomor Surat")

    # =========================
    # 🔥 SIMPAN INFO PERUSAHAAN (INI YANG KAMU CARI)
    # =========================
    st.session_state.company_info = {
        "Nama Perusahaan": nama,
        "Alamat Perusahaan": alamat,
        "Nomor Surat": nomor
    }

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
    # 🔥 PREVIEW
    # =========================
    if st.button("Preview Data"):

        parsed_rows = []

        for d in data_list:

            koordinat = d.get("koordinat", "")
            tanggal = d.get("tanggal", "")

            parsed = parse_coordinate(koordinat)

            if parsed is None:
                st.error(f"Koordinat tidak valid: {koordinat}")
                st.stop()

            # simpan parsed terakhir
            st.session_state.last_parsed = parsed

            parsed_rows.append({
                "Tanggal Koordinat": str(tanggal),
                "Koordinat": koordinat,
            
                # 🔥 TAMBAHAN
                "Koordinat Awal": koordinat,
                "Koordinat Akhir": koordinat,
            
                "Koordinat Awal (Desimal)": parsed.get("Koordinat Awal (Desimal)"),
                "Koordinat Akhir (Desimal)": parsed.get("Koordinat Akhir (Desimal)"),
            
                "Mode": parsed.get("mode"),
                "All Points": parsed.get("All Points", [])
            })
        df_preview = pd.DataFrame(parsed_rows)
        st.session_state.preview_data = df_preview
        

        # 🔥 FIX PENTING (INI YANG HILANG)
        st.session_state.selected_data = df_preview

        st.success("✅ Data berhasil diparsing")
        st.dataframe(df_preview)

# =========================
# 🔥 PREVIEW + MAP (GLOBAL) — FINAL FIX
# =========================

# 🔥 AMBIL DATA DARI selected_data (UTAMA)
df_preview = st.session_state.get("selected_data")

# 🔥 FALLBACK KE preview_data (KHUSUS MANUAL)
if df_preview is None:
    df_preview = st.session_state.get("preview_data")

if df_preview is not None:

    st.success("✅ Data berhasil diparsing")
    st.dataframe(df_preview)

    import folium
    from streamlit_folium import st_folium

    st.subheader("🗺️ Preview Lokasi")

    try:
        # =========================
        # 🔵 MODE TITIK
        # =========================
        if len(df_preview) == 1 and df_preview.iloc[0]["Mode"] == "titik":

            latlon = df_preview.iloc[0]["Koordinat Awal (Desimal)"]

            # 🔥 AMAN (HANDLE STRING ATAU TUPLE)
            if isinstance(latlon, str):
                lat, lon = map(float, latlon.split(","))
            else:
                lat, lon = latlon

            st.success(f"📍 Titik otomatis: {lat}, {lon}")

            m = folium.Map(location=[lat, lon], zoom_start=7)

            folium.Marker([lat, lon]).add_to(m)

            st_folium(m, height=400)

        # =========================
        # 🟢 MODE RUTE
        # =========================
        else:
            all_points = []

            for _, row in df_preview.iterrows():
                for lat, lon in row.get("All Points", []):
                    all_points.append((lat, lon))

            if all_points:
                center_lat = sum(p[0] for p in all_points) / len(all_points)
                center_lon = sum(p[1] for p in all_points) / len(all_points)

                m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

                for lat, lon in all_points:
                    folium.Marker([lat, lon]).add_to(m)

                # garis rute
                if len(all_points) > 1:
                    folium.PolyLine(all_points).add_to(m)

                st_folium(m, height=400)

    except Exception as e:
        st.warning("Map error")
        st.exception(e)

    # =========================
    # SAVE (KHUSUS MANUAL — TIDAK DIUBAH)
    # =========================
    if st.session_state.get("preview_data") is not None:

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

    if not st.session_state.get("manual_saved", False):
        pass  # 🔥 jangan stop global (biar Google Sheet tetap jalan)

# =========================
# 🔥 FIX STATE df_id (WAJIB)
# =========================
if df_id is None and st.session_state.get("df_id_manual") is not None:
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

    if st.session_state.get("results_module5") is None:
        st.warning("Jalankan analisis dulu")
        st.stop()

    # =========================
    # 🔥 FIX: BUILD module1_rows (UNIFIED)
    # =========================
    module1_rows = []

    # ambil data perusahaan (khusus manual)
    company_info = st.session_state.get("company_info", {})

    df_data = st.session_state.get("selected_data")

    # 🔥 AUTO FALLBACK
    if df_data is None:
        df_data = st.session_state.get("preview_data")
    
    if df_data is None or df_data.empty:
        st.warning("Data belum tersedia. Klik 'Preview Data' dulu.")
        st.stop()
    
    for row in df_data.to_dict("records"):

        new_row = row.copy()

        # =========================
        # 🔥 INJECT DATA PERUSAHAAN
        # =========================
        new_row["Nama Perusahaan"] = company_info.get(
            "Nama Perusahaan",
            row.get("Nama Perusahaan", "")
        )

        new_row["Alamat Perusahaan"] = company_info.get(
            "Alamat Perusahaan",
            row.get("Alamat Perusahaan", "")
        )

        new_row["Nomor Surat"] = company_info.get(
            "Nomor Surat",
            row.get("Nomor Surat", "")
        )

        module1_rows.append(new_row)

    # =========================
    # GENERATE DOCX
    # =========================
    buffer = generate_final_docx_streamlit(
        module1_rows,
        st.session_state.results_module5,
        template_path
    )

    st.download_button(
        "📥 Download Laporan",
        buffer,
        file_name="PTSP_Report.docx"
    )
