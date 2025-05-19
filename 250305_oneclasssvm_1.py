import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

def segment_beads(df, column, threshold):
    start_indices, end_indices = [], []
    signal = df[column].to_numpy()
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))

def extract_features(signal):
    if len(signal) == 0:
        return [0] * 10
    return [
        np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
        skew(signal), kurtosis(signal),
        np.sum(signal ** 2), np.sqrt(np.mean(signal ** 2)),
        len(find_peaks(signal)[0]), np.sum(np.diff(np.sign(signal)) != 0)
    ]

st.title("Laser Welding Normality Detection")

with st.sidebar:
    uploaded_zip = st.file_uploader("Upload a ZIP file with normal data", type=["zip"])
    if uploaded_zip:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_zip.getbuffer())
        csv_files = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")
        
        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)
        
        if st.button("Segment Beads"):
            bead_metadata = []
            for file in csv_files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    bead_metadata.append({"file": file, "bead_number": bead_num, "start": start, "end": end})
            st.session_state["metadata"] = bead_metadata
            st.success("Bead segmentation complete")
        
        bead_numbers = st.text_input("Enter bead numbers (comma-separated)")
        if st.button("Select Beads"):
            selected_beads = [int(b.strip()) for b in bead_numbers.split(",") if b.strip().isdigit()]
            st.session_state["selected_beads"] = selected_beads
            st.success("Beads selected successfully")

        if st.button("Start Model Training") and "metadata" in st.session_state:
            bead_features = []
            for entry in st.session_state["metadata"]:
                if entry["bead_number"] in st.session_state["selected_beads"]:
                    df = pd.read_csv(entry["file"])
                    signal = df.iloc[entry["start"]:entry["end"] + 1][filter_column].values
                    bead_features.append(extract_features(signal))
            if bead_features:
                scaler = RobustScaler()
                X_train = scaler.fit_transform(bead_features)
                model = OneClassSVM()
                model.fit(X_train)
                st.session_state["model"] = model
                st.session_state["scaler"] = scaler
                st.success("Model trained successfully")
                
if "model" in st.session_state:
    new_csv = st.file_uploader("Upload a new CSV file for analysis", type=["csv"])
    if new_csv:
        df_new = pd.read_csv(new_csv)
        segments_new = segment_beads(df_new, filter_column, threshold)
        bead_results = {}
        for bead_num, (start, end) in enumerate(segments_new, start=1):
            if bead_num in st.session_state["selected_beads"]:
                signal = df_new.iloc[start:end + 1][filter_column].values
                features = np.array(extract_features(signal)).reshape(1, -1)
                features_scaled = st.session_state["scaler"].transform(features)
                prediction = st.session_state["model"].predict(features_scaled)
                bead_results[bead_num] = "Normal" if prediction[0] == 1 else "Anomaly"
        
        for bead_num, status in bead_results.items():
            st.write(f"Bead {bead_num}: {status}")
