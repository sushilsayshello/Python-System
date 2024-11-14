# Import necessary libraries
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2

# Load the dataset (upload functionality in Streamlit)
st.title("DDoS Detection and Network Traffic Analysis Application")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data based on file type
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    # Standardize column names
    data.columns = data.columns.str.strip().str.lower()
    st.write("Columns in the dataset after processing:", data.columns.tolist())

    # Dataset Overview with interactive table
    st.subheader("Dataset Overview")
    st.dataframe(data.head())

    # Label Distribution
    st.subheader("Label Distribution")
    if 'label' in data.columns:
        label_counts = data['label'].value_counts()
        label_chart = px.bar(label_counts, x=label_counts.index, y=label_counts.values, labels={'x': 'Label', 'y': 'Count'})
        label_chart.update_layout(title="Label Distribution")
        st.plotly_chart(label_chart)
    else:
        st.error("Error: 'label' column not found in the dataset.")
        st.stop()

    # Data Preprocessing
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    st.write("Label Encoding Mapping:", label_mapping)

    # Key Feature Analysis
    st.subheader("Key Feature Analysis")

    # Select 10 relevant features
    features_to_analyze = [
        'pktcount', 'bytecount', 'dur', 'tot_dur', 'flows', 'pktperflow', 
        'byteperflow', 'protocol', 'port_no', 'pktrate'
    ]

    # Display summary statistics for the selected features
    st.write("Summary Statistics for Key Features:")
    st.dataframe(data[features_to_analyze].describe())

    # Visualizations for Key Features
    st.subheader("Feature Visualizations")

    # 1. Packet Count Distribution
    st.write("**Packet Count Distribution**")
    pktcount_fig = px.histogram(data, x='pktcount', title="Packet Count Distribution")
    st.plotly_chart(pktcount_fig)

    # 2. Byte Count Distribution
    st.write("**Byte Count Distribution**")
    bytecount_fig = px.histogram(data, x='bytecount', title="Byte Count Distribution")
    st.plotly_chart(bytecount_fig)

    # 3. Duration Analysis
    st.write("**Flow Duration Distribution**")
    dur_fig = px.histogram(data, x='dur', title="Flow Duration Distribution")
    st.plotly_chart(dur_fig)

    # 4. Total Duration Analysis
    st.write("**Total Duration Distribution**")
    tot_dur_fig = px.histogram(data, x='tot_dur', title="Total Duration Distribution")
    st.plotly_chart(tot_dur_fig)

    # 5. Flows Count Analysis
    st.write("**Flows Count Analysis**")
    flows_fig = px.histogram(data, x='flows', title="Flows Count Distribution")
    st.plotly_chart(flows_fig)

    # 6. Packets Per Flow
    st.write("**Packets Per Flow Distribution**")
    pktperflow_fig = px.histogram(data, x='pktperflow', title="Packets Per Flow Distribution")
    st.plotly_chart(pktperflow_fig)

    # 7. Bytes Per Flow
    st.write("**Bytes Per Flow Distribution**")
    byteperflow_fig = px.histogram(data, x='byteperflow', title="Bytes Per Flow Distribution")
    st.plotly_chart(byteperflow_fig)

    # 8. Protocol Distribution
    if 'protocol' in data.columns:
        st.write("**Protocol Distribution**")
        protocol_counts = data['protocol'].value_counts()
        protocol_fig = px.pie(protocol_counts, values=protocol_counts.values, names=protocol_counts.index, title="Protocol Distribution")
        st.plotly_chart(protocol_fig)

    # 9. Port Number Analysis
    if 'port_no' in data.columns:
        st.write("**Port Number Analysis**")
        port_counts = data['port_no'].value_counts().nlargest(10)
        port_fig = px.bar(port_counts, x=port_counts.index, y=port_counts.values, labels={'x': 'Port Number', 'y': 'Count'}, title="Top 10 Port Numbers")
        st.plotly_chart(port_fig)

    # 10. Packet Rate Analysis
    st.write("**Packet Rate Distribution**")
    pktrate_fig = px.histogram(data, x='pktrate', title="Packet Rate Distribution")
    st.plotly_chart(pktrate_fig)

    # Model Training for Attack Detection
    st.subheader("DDoS Attack Detection Model")

    # Feature Selection and Model Training
    X = data[features_to_analyze]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_labels = list(label_mapping.keys())
    cm_fig = px.imshow(cm, text_auto=True, labels={'x': "Predicted", 'y': "Actual"}, x=cm_labels, y=cm_labels)
    cm_fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(cm_fig)

    # Time-Series Analysis (if timestamp column is available)
    if 'dt' in data.columns:
        data['dt'] = pd.to_datetime(data['dt'], errors='coerce')
        data = data.dropna(subset=['dt'])
        st.write("**Traffic Trends Over Time**")
        
        time_chart = alt.Chart(data).mark_line().encode(
            x='dt:T', y='pktcount:Q', color='label:N',
            tooltip=['dt:T', 'pktcount:Q', 'label:N']
        ).properties(title="Packet Count Over Time by Label")
        
        st.altair_chart(time_chart, use_container_width=True)

        st.write("**Total Byte Count Over Time**")
        byte_time_chart = alt.Chart(data).mark_line().encode(
            x='dt:T', y='bytecount:Q', color='label:N',
            tooltip=['dt:T', 'bytecount:Q', 'label:N']
        ).properties(title="Byte Count Over Time by Label")
        
        st.altair_chart(byte_time_chart, use_container_width=True)
    else:
        st.warning("Timestamp column 'dt' is missing or improperly formatted. Time-series analysis is disabled.")
