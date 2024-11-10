# Step 1: Import necessary libraries
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import datetime as dt
import random

# Step 2: Load the dataset (upload functionality in Streamlit)
st.title("Comprehensive Network Flow Simulation with Real-Time DDoS Attack Alerts and Website Vulnerability Scanner")
uploaded_file = st.file_uploader("Upload your dataset", type=["xlsx"])

# Step 3: Protection mode toggle
protection_mode = st.sidebar.checkbox("Enable Protection Mode")
if protection_mode:
    st.sidebar.success("Protection Mode is ON")
else:
    st.sidebar.warning("Protection Mode is OFF")

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    
    # Normalize column names to lowercase and strip spaces
    data.columns = data.columns.str.strip().str.lower()

    # Display the columns for debugging purposes
    st.write("Columns in the dataset after processing:", data.columns.tolist())

    # Step 4: Data exploration
    st.subheader("Dataset Overview")
    st.dataframe(data.head())

    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Display label distribution
    st.subheader("Label Distribution")
    if 'label' in data.columns:
        label_counts = data['label'].value_counts()
        st.bar_chart(label_counts)
    else:
        st.error("Error: 'label' column not found in the dataset.")
        st.stop()

    # Step 5: Add interactive filter widgets
    st.sidebar.header("Filter Options")
    unique_labels = data['label'].unique()
    selected_labels = st.sidebar.multiselect("Select Labels to View", options=unique_labels, default=unique_labels)
    filtered_data = data[data['label'].isin(selected_labels)]

    # Step 6: Data preprocessing
    label_encoder = LabelEncoder()
    filtered_data['label'] = label_encoder.fit_transform(filtered_data['label'])

    # Display mapping of encoded labels
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    st.write("Label Encoding Mapping:", label_mapping)

    # Ensure the 'timestamp' column is in datetime format
    if 'timestamp' in filtered_data.columns:
        filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'], errors='coerce')
        filtered_data = filtered_data.dropna(subset=['timestamp'])

    # Select features and target
    X = filtered_data.drop(columns=['label', 'flow id', 'timestamp'], errors='ignore')  # Drop non-feature columns as needed
    y = filtered_data['label']

    # Ensure only numeric columns are used for scaling
    numeric_cols = X.select_dtypes(include=['number']).columns
    X_numeric = X[numeric_cols]

    # Replace inf and NaN values
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    X_numeric = X_numeric.dropna()  # Optionally, fill with X_numeric.fillna(X_numeric.mean())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y.loc[X_numeric.index], test_size=0.3, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 7: Train an XGBoost classifier with real-time updates
    clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    st.subheader("Training the XGBoost Classifier")
    with st.spinner("Training in progress..."):
        for i in range(1, 101):
            # Simulate training progress
            time.sleep(0.05)
            st.progress(i)
        clf.fit(X_train, y_train)

    # Step 8: Make predictions and evaluate the model
    y_pred = clf.predict(X_test)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, ax=ax_cm, annot=True, fmt='d', cmap='Blues')
    ax_cm.set_title('Confusion Matrix')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)

    st.subheader("Accuracy Score")
    st.write(accuracy_score(y_test, y_pred))

    # Display count of benign and attack labels
    st.subheader("Detailed Label Counts")
    st.write("Benign Count:", (y_test == label_mapping.get('benign', -1)).sum())
    st.write("Attack Count:", (y_test != label_mapping.get('benign', -1)).sum())

    # Step 9: Time-series visualization of attacks
    st.subheader("Time-Series Visualization of Attacks")
    attack_data = filtered_data[filtered_data['label'] != label_mapping.get('benign', -1)]
    benign_data = filtered_data[filtered_data['label'] == label_mapping.get('benign', -1)]

    fig_time_series, ax_time_series = plt.subplots()
    ax_time_series.plot(attack_data['timestamp'], attack_data['label'], 'r.', label='Attack')
    ax_time_series.plot(benign_data['timestamp'], benign_data['label'], 'g.', label='Benign')
    ax_time_series.set_title('Time-Series Visualization of Benign and Attack Traffic')
    ax_time_series.set_xlabel('Timestamp')
    ax_time_series.set_ylabel('Label')
    ax_time_series.legend()
    st.pyplot(fig_time_series)

    # Step 10: Real-time DDoS attack alerts
    st.subheader("Real-Time DDoS Attack Alerts")
    latest_attack_time = attack_data['timestamp'].max()
    if pd.notnull(latest_attack_time):
        time_since_last_attack = dt.datetime.now() - latest_attack_time
        if time_since_last_attack < dt.timedelta(minutes=5):
            st.error(f"ALERT: DDoS attack detected at {latest_attack_time}! Immediate action required.")
        else:
            st.success(f"No recent DDoS attacks detected. Last recorded attack was at {latest_attack_time}.")
    else:
        st.success("No DDoS attacks detected in the current dataset.")

    # Step 11: Simulate network traffic and add alerts
    st.subheader("Simulated Network Traffic")
    simulate_traffic = st.button("Start Traffic Simulation")
    if simulate_traffic:
        for _ in range(50):
            traffic_type = random.choice(['benign', 'attack'])
            traffic_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"Simulated traffic detected: {traffic_type} at {traffic_time}")
            if protection_mode and traffic_type == 'attack':
                st.error(f"ALERT: Attack traffic detected at {traffic_time} under Protection Mode!")
            time.sleep(0.2)

    # Step 12: Plot real-time graph of predictions
    st.subheader("Real-Time Simulation of Model Predictions")
    fig, ax = plt.subplots()
    ax.plot(range(len(y_test)), y_test, label='Actual', color='blue')
    ax.plot(range(len(y_pred)), y_pred, label='Predicted', color='red', linestyle='--')
    ax.set_title('Actual vs Predicted Labels')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Label')
    ax.legend()
    st.pyplot(fig)

    # Step 13: Add data download button
    st.sidebar.subheader("Download Processed Data")
    processed_data_csv = filtered_data.to_csv(index=False)
    st.sidebar.download_button(label="Download CSV", data=processed_data_csv, file_name='processed_data.csv', mime='text/csv')

# Step 14: Website Vulnerability Scanner
def check_vulnerabilities(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            st.write(f"**Scanning {url}**")
            soup = BeautifulSoup(response.text, 'html.parser')
            forms = soup.find_all('form')
            st.write(f"Number of forms detected: {len(forms)}")
            for form in forms:
                st.write("Form detected:")
                st.write(form.prettify())
                st.write("Potential vulnerability: Ensure inputs are sanitized to avoid XSS and injection attacks.")
        else:
            st.write(f"Failed to access {url}. Status code: {response.status_code}")
    except Exception as e:
        st.write(f"An error occurred while scanning: {e}")

st.subheader("Website Vulnerability Scanner")
url = st.text_input("Enter a URL to scan for vulnerabilities")
if url:
    check_vulnerabilities(url)
else:
    st.write("Please enter a URL to begin scanning.")

