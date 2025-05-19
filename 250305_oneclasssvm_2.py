import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew, kurtosis
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
    """Extracts a consistent set of features from the given signal."""
    if len(signal) == 0:
        return [0] * 10  # Ensure consistent feature length
    return [
        np.mean(signal), np.std(signal), np.min(signal), np.max(signal),
        skew(signal), kurtosis(signal), np.sum(signal**2),
        np.sqrt(np.mean(signal**2)), len(find_peaks(signal)[0]),
        np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
    ]

st.set_page_config(layout="wide")
st.title("Laser Welding One-Class SVM")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a ZIP file containing normal data CSV files", type=["zip"])
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
            with st.spinner("Segmenting beads..."):
                bead_segments = {file: segment_beads(pd.read_csv(file), filter_column, threshold) for file in csv_files}
                st.session_state["bead_segments"] = bead_segments
                st.success("Bead segmentation complete")
        
        bead_numbers = st.text_input("Enter bead numbers (comma-separated)")
        if st.button("Select Beads") and "bead_segments" in st.session_state:
            selected_beads = [int(b.strip()) for b in bead_numbers.split(",") if b.strip().isdigit()]
            st.session_state["selected_beads"] = selected_beads
            st.success("Beads selected successfully!")
        
        if st.button("Start Model Training") and "selected_beads" in st.session_state:
            with st.spinner("Training One-Class SVM..."):
                training_data = []
                for file, segments in st.session_state["bead_segments"].items():
                    df = pd.read_csv(file)
                    for bead_num, (start, end) in enumerate(segments, start=1):
                        if bead_num in st.session_state["selected_beads"]:
                            features = extract_features(df.iloc[start:end + 1, 0].values)
                            training_data.append(features)
                
                training_data = np.array(training_data)
                
                scaler = RobustScaler()
                training_data_scaled = scaler.fit_transform(training_data)
                
                oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale', shrinking=True)
                oc_svm.fit(training_data_scaled)
                
                st.session_state.update({"model": oc_svm, "scaler": scaler, "training_beads": bead_numbers})
                st.success("Model training complete!")

new_csv = st.file_uploader("Upload a new CSV file for evaluation", type=["csv"])
if new_csv and "model" in st.session_state:
    with st.spinner("Processing new data..."):
        df = pd.read_csv(new_csv)
        segments = segment_beads(df, filter_column, threshold)
        test_data, test_signals, test_beads = [], [], []
        for bead_num, (start, end) in enumerate(segments, start=1):
            if bead_num in st.session_state["selected_beads"]:
                signal = df.iloc[start:end + 1, 0].values
                test_data.append(extract_features(signal))
                test_signals.append(signal)
                test_beads.append(bead_num)
        
        test_data = np.array(test_data)
        test_data_scaled = st.session_state["scaler"].transform(test_data)
        predictions = st.session_state["model"].predict(test_data_scaled)
        
        st.session_state.update({"test_results": dict(zip(test_beads, predictions)), "test_signals": dict(zip(test_beads, test_signals))})
        st.success("Evaluation complete!")

st.write("## Visualization")
if "test_results" in st.session_state and "test_signals" in st.session_state:
    for bead_number, prediction in st.session_state["test_results"].items():
        fig = go.Figure()
        for file in csv_files:
            df = pd.read_csv(file)
            for start, end in st.session_state["bead_segments"].get(file, []):
                if bead_number in st.session_state["selected_beads"]:
                    fig.add_trace(go.Scatter(y=df.iloc[start:end + 1, 0].values, mode='lines', line=dict(color='black', width=1), name=f"Training: {file}"))
        
        color = 'red' if prediction == -1 else 'blue'
        fig.add_trace(go.Scatter(y=st.session_state["test_signals"][bead_number], mode='lines', line=dict(color=color, width=1.5), name=f"Test: Bead {bead_number}"))
        
        fig.update_layout(title=f"Bead {bead_number}: One-Class SVM", xaxis_title="Time Index", yaxis_title="Signal Value", showlegend=True)
        st.plotly_chart(fig)

st.success("Analysis complete!")
