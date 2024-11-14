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

    # Filtering and Feature Selection
    st.sidebar.subheader("Feature Selection and Filtering")
    features_to_analyze = st.sidebar.multiselect("Select Features for Analysis", options=data.columns.tolist(), default=[
        'pktcount', 'bytecount', 'dur', 'tot_dur', 'flows', 'pktperflow', 
        'byteperflow', 'protocol', 'port_no', 'pktrate'
    ])
    filtered_data = data[features_to_analyze + ['label']].copy()

    # Handle non-numeric data and missing values for model training
    X = filtered_data.drop(columns=['label']).select_dtypes(include=['number'])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())  # Fill missing values with mean of each column
    y = filtered_data['label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train XGBoost Classifier
    st.subheader("Training the XGBoost Classifier")
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

    # Visualizations for Selected Features
    st.subheader("Feature Visualizations")

    for feature in features_to_analyze:
        if feature in filtered_data.columns:
            st.write(f"**{feature.capitalize()} Distribution**")
            fig = px.histogram(filtered_data, x=feature, title=f"{feature.capitalize()} Distribution")
            st.plotly_chart(fig)

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
