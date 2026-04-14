# =========================
# MODULE 1 – GOOGLE SHEET VIA SERVICE ACCOUNT (FINAL FIXED)
# =========================

import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from pathlib import Path
import datetime

# =========================
# REQUIRED COLUMNS
# =========================
REQUIRED_COLUMNS = [
    "Id",
    "Requester",
    "Timestamp",
    "Nama Perusahaan",
    "Alamat Perusahaan",
    "Nomor Surat",
    "Informasi",
    "Tanggal Koordinat",
    "Koordinat",
    "Koordinat Awal",
    "Koordinat Akhir",
    "Koordinat Awal (Desimal)",
    "Koordinat Akhir (Desimal)",
    "Water Checker Awal",
    "Water Checker Akhir"
]

# =========================
# VALIDATOR
# =========================
def validate_request_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    return df.reset_index(drop=True)


# =========================
# AUTH CLIENT
# =========================
def get_gspread_client():

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    if Path("service_account.json").exists():
        creds = Credentials.from_service_account_file(
            "service_account.json",
            scopes=scopes,
        )
    else:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scopes,
        )

    return gspread.authorize(creds)


# =========================
# CONFIG
# =========================
def get_sheet_config():
    config = st.secrets["google_sheet"]

    return {
        "spreadsheet_id": config["spreadsheet_id"],
        "sheet_n8n": config["worksheet_n8n"],
        "sheet_manual": config["worksheet_manual"],
    }


# =========================
# LOAD GENERIC SHEET
# =========================
def load_sheet(sheet_name):

    try:
        client = get_gspread_client()
        cfg = get_sheet_config()

        sheet = client.open_by_key(cfg["spreadsheet_id"]).worksheet(sheet_name)

        data = sheet.get_all_records()
        df = pd.DataFrame(data)

        if df.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        return validate_request_dataframe(df)

    except Exception as e:
        st.warning(f"Gagal load sheet {sheet_name}: {e}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)


# =========================
# LOAD N8N (SURAT)
# =========================
@st.cache_data(show_spinner=False)
def load_google_sheet():
    cfg = get_sheet_config()
    return load_sheet(cfg["sheet_n8n"])


# =========================
# LOAD MANUAL (INPUT_MANUAL)
# =========================
@st.cache_data(show_spinner=False)
def load_manual_sheet():
    cfg = get_sheet_config()
    df = load_sheet(cfg["sheet_manual"])

    if df.empty:
        st.warning("Sheet Input_Manual kosong atau gagal dibaca")

    return df


# =========================
# 🔥 NEW: GET LIST ID MANUAL (INI YANG DIPAKAI DI UI)
# =========================
def get_manual_id_list():

    df = load_manual_sheet()

    if df.empty or "Id" not in df.columns:
        return []

    return sorted(df["Id"].dropna().astype(str).unique())


# =========================
# GET DATA BY ID (PRIORITAS MANUAL)
# =========================
def get_data_by_id(id_surat):

    df_manual = load_manual_sheet()
    df_n8n = load_google_sheet()

    # 🔥 PRIORITAS MANUAL
    row = df_manual[df_manual["Id"].astype(str) == str(id_surat)]
    if not row.empty:
        return row.iloc[0].to_dict()

    # fallback ke n8n
    row = df_n8n[df_n8n["Id"].astype(str) == str(id_surat)]
    if not row.empty:
        return row.iloc[0].to_dict()

    return None


# =========================
# GENERATE ID (MANUAL)
# =========================
def generate_id():

    try:
        df = load_manual_sheet()

        if df.empty or "Id" not in df.columns:
            return "PTSP-001"

        ids = df["Id"].dropna().astype(str)

        numbers = []

        for i in ids:
            if i.startswith("PTSP-"):
                try:
                    numbers.append(int(i.split("-")[1]))
                except:
                    continue

        next_id = max(numbers) + 1 if numbers else 1

        return f"PTSP-{str(next_id).zfill(3)}"

    except Exception as e:
        st.warning(f"Gagal generate ID: {e}")
        return "PTSP-001"


# =========================
# SAVE MANUAL INPUT
# =========================
def save_manual_input(data):

    client = get_gspread_client()
    cfg = get_sheet_config()

    sheet = client.open_by_key(cfg["spreadsheet_id"]).worksheet(cfg["sheet_manual"])

    sheet.append_row([
        data.get("Id", ""),
        data.get("Requester", ""),
        data.get("Timestamp", ""),
        data.get("Nama Perusahaan", ""),
        data.get("Alamat Perusahaan", ""),
        data.get("Nomor Surat", ""),
        data.get("Informasi", ""),
        data.get("Tanggal Koordinat", ""),
        data.get("Koordinat", ""),
        data.get("Koordinat Awal", ""),
        data.get("Koordinat Akhir", ""),
        data.get("Koordinat Awal (Desimal)", ""),
        data.get("Koordinat Akhir (Desimal)", ""),
        data.get("Water Checker Awal", ""),
        data.get("Water Checker Akhir", "")
    ])


# =========================
# DISPLAY (N8N ONLY)
# =========================
def load_request_sheet_streamlit():

    st.subheader("📄 Data Permintaan PTSP (Surat / Telegram)")

    try:
        df = load_google_sheet()
        st.success("✅ Data Google Sheet berhasil dimuat")
        st.write(f"Total permintaan: **{len(df)}**")
        return df

    except Exception as e:
        st.error("❌ Gagal memuat Google Sheet")
        st.exception(e)
        return None
