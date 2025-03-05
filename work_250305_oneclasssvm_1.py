import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import numpy as np

def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    return [os.path.join(extract_dir, f) for f in csv_files]

def segment_beads(df, column, threshold):
    start_indices = []
    end_indices = []
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

def extract_advanced_features(signal):
    if len(signal) == 0:
        return [0] * 20
    
    features = [
        np.mean(signal), np.std(signal), np.min(signal), np.max(signal), np.median(signal),
        skew(signal), kurtosis(signal), np.ptp(signal), np.sum(signal**2),
        np.std(signal) / np.mean(signal) if np.mean(signal) != 0 else 0
    ]
    
    signal_fft = fft(signal)
    psd = np.abs(signal_fft)**2
    freqs = fftfreq(len(signal), 1)[:len(signal)//2]
    psd_norm = psd[:len(signal)//2] / np.sum(psd[:len(signal)//2])
    
    features.extend([
        freqs[np.argmax(psd_norm)] if len(psd_norm) > 0 else 0,
        -np.sum(psd_norm * np.log2(psd_norm + 1e-12)),
        np.corrcoef(signal[:-1], signal[1:])[0, 1] if len(signal) > 1 else 0,
        len(find_peaks(signal)[0]), np.sum(np.diff(np.sign(signal)) != 0) / len(signal),
        np.sqrt(np.mean(signal**2)), np.polyfit(np.arange(len(signal)), signal, 1)[0]
    ])
    
    return features

st.title("Laser Welding Anomaly Detection with One-Class SVM")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload OK Data (ZIP)", type=["zip"])
    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_files = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")
        
        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)
        
        if st.button("Segment Beads"):
            bead_segments = {}
            metadata = []
            for file in csv_files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    metadata.append({"file": file, "bead_number": bead_num, "start": start, "end": end})
            st.session_state["metadata"] = metadata
            st.success("Bead segmentation complete")
        
        if st.button("Train One-Class SVM") and "metadata" in st.session_state:
            feature_data = {}
            for entry in st.session_state["metadata"]:
                df = pd.read_csv(entry["file"])
                signal = df.iloc[entry["start"]:entry["end"] + 1, 0].values
                features = extract_advanced_features(signal)
                feature_data.setdefault(entry["bead_number"], []).append(features)
            
            st.session_state["scalers"] = {}
            st.session_state["svm_models"] = {}
            for bead_num, feature_matrix in feature_data.items():
                feature_matrix = np.array(feature_matrix)
                scaler = RobustScaler().fit(feature_matrix)
                st.session_state["scalers"][bead_num] = scaler
                feature_matrix = scaler.transform(feature_matrix)
                model = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto').fit(feature_matrix)
                st.session_state["svm_models"][bead_num] = model
            st.success("Model training complete!")
        
    test_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])
    if test_file and "svm_models" in st.session_state:
        df_test = pd.read_csv(test_file)
        test_metadata = segment_beads(df_test, filter_column, threshold)
        
        fig = go.Figure()
        for bead_num, (start, end) in enumerate(test_metadata, start=1):
            signal = df_test.iloc[start:end + 1, 0].values
            features = extract_advanced_features(signal)
            if bead_num in st.session_state["svm_models"]:
                model = st.session_state["svm_models"][bead_num]
                scaler = st.session_state["scalers"][bead_num]
                features = scaler.transform([features])
                prediction = model.predict(features)[0]
                color = "black" if prediction == 1 else "red"
                fig.add_trace(go.Scatter(y=signal, mode='lines', line=dict(color=color, width=1), name=f"Bead {bead_num}"))
        st.plotly_chart(fig)
