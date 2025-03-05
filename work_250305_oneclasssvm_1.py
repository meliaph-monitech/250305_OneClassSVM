import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
import numpy as np

# Function to extract ZIP files
def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    return [os.path.join(extract_dir, f) for f in csv_files], extract_dir

# Function for bead segmentation
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

# Feature extraction
def extract_features(signal):
    if len(signal) == 0:
        return [0] * 5
    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.median(signal)
    ]

st.title("Laser Welding Normal Data Learning & Anomaly Detection")

# Upload ZIP file
uploaded_file = st.file_uploader("Upload a ZIP file containing normal data CSV files", type=["zip"])
if uploaded_file:
    with open("temp.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    csv_files, extract_dir = extract_zip("temp.zip")
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
            if segments:
                bead_segments[file] = segments
                for bead_num, (start, end) in enumerate(segments, start=1):
                    metadata.append({"file": file, "bead_number": bead_num, "start_index": start, "end_index": end})
        st.session_state["metadata"] = metadata
        st.success("Bead segmentation complete")
    
    bead_numbers = st.text_input("Enter bead numbers (comma-separated)")
    if st.button("Select Beads") and "metadata" in st.session_state:
        selected_beads = [int(b.strip()) for b in bead_numbers.split(",") if b.strip().isdigit()]
        chosen_bead_data = []
        for entry in st.session_state["metadata"]:
            if entry["bead_number"] in selected_beads:
                df = pd.read_csv(entry["file"])
                bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                chosen_bead_data.append({"data": bead_segment, "file": entry["file"], "bead_number": entry["bead_number"]})
        st.session_state["chosen_bead_data"] = chosen_bead_data
        st.success("Beads selected successfully!")

# Train One-Class SVM Model
if st.button("Start Model Training") and "chosen_bead_data" in st.session_state:
    feature_matrix = np.array([extract_features(seg["data"].iloc[:, 0].values) for seg in st.session_state["chosen_bead_data"]])
    scaler = RobustScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    model = OneClassSVM(nu=0.01, kernel='rbf', gamma='auto')
    model.fit(feature_matrix)
    st.session_state["scaler"] = scaler
    st.session_state["model"] = model
    st.success("Model training complete!")

# Upload new CSV for analysis
if "model" in st.session_state:
    st.subheader("Upload New CSV for Analysis")
    new_file = st.file_uploader("Upload a new welding process CSV", type=["csv"])
    if new_file:
        df_new = pd.read_csv(new_file)
        new_segments = segment_beads(df_new, filter_column, threshold)
        
        if new_segments:
            results = {}
            for bead_num, (start, end) in enumerate(new_segments, start=1):
                if bead_num in selected_beads:
                    bead_data = df_new.iloc[start:end + 1, 0].values
                    features = np.array(extract_features(bead_data)).reshape(1, -1)
                    features = st.session_state["scaler"].transform(features)
                    prediction = st.session_state["model"].predict(features)
                    results[bead_num] = "normal" if prediction[0] == 1 else "not normal"
            
            # Visualization
            for bead_number in selected_beads:
                fig = go.Figure()
                for seg in st.session_state["chosen_bead_data"]:
                    if seg["bead_number"] == bead_number:
                        fig.add_trace(go.Scatter(y=seg["data"].iloc[:, 0].values, mode='lines', line=dict(color='black'), name=f"Training - {seg['file']}"))
                if bead_number in results:
                    color = 'red' if results[bead_number] == "not normal" else 'blue'
                    fig.add_trace(go.Scatter(y=df_new.iloc[new_segments[bead_number - 1][0]:new_segments[bead_number - 1][1] + 1, 0].values, mode='lines', line=dict(color=color), name=f"New Data - {results[bead_number]}"))
                fig.update_layout(title=f"Bead Number {bead_number}", xaxis_title="Time Index", yaxis_title="Signal Value")
                st.plotly_chart(fig)
            st.success("Analysis complete!")
