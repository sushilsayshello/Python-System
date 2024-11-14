# Import necessary libraries
import pandas as pd
import streamlit as st
import numpy as np
import datetime as dt
import plotly.express as px
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2

# Load the dataset (upload functionality in Streamlit)
st.title("DDoS Detection and Analytics Application")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

# Protection mode toggle
protection_mode = st.sidebar.checkbox("Enable Protection Mode")
if protection_mode:
    st.sidebar.success("Protection Mode is ON")
else:
    st.sidebar.warning("Protection Mode is OFF")

if uploaded_file is not None:
    # Load data based on file type
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    data.columns = data.columns.str.strip().str.lower()
    st.write("Columns in the dataset after processing:", data.columns.tolist())

    # Dataset Overview with interactive table
    st.subheader("Dataset Overview")
    st.dataframe(data.head())

    # Label Distribution with Plotly
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
    unique_labels = data['label'].unique()
    selected_labels = st.sidebar.multiselect("Select Labels to View", options=unique_labels, default=unique_labels)
    filtered_data = data[data['label'].isin(selected_labels)]
    label_encoder = LabelEncoder()
    filtered_data['label'] = label_encoder.fit_transform(filtered_data['label'])
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    st.write("Label Encoding Mapping:", label_mapping)

    # Feature Selection
    st.sidebar.subheader("Feature Selection")
    num_features = st.sidebar.slider("Select number of features", min_value=1, max_value=len(filtered_data.columns) - 1, value=5)
    X = filtered_data.drop(columns=['label', 'dt'], errors='ignore')
    y = filtered_data['label']

    # Select K best features
    X_numeric = X.select_dtypes(include=['number'])
    selector = SelectKBest(chi2, k=num_features).fit(X_numeric, y)
    selected_features = X_numeric.columns[selector.get_support()]
    st.write("Selected Features:", selected_features.tolist())
    X_selected = X_numeric[selected_features]

    # Replace inf and NaN values
    X_selected = X_selected.replace([np.inf, -np.inf], np.nan)
    X_selected = X_selected.fillna(X_selected.mean())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train XGBoost Classifier
    st.subheader("Training the XGBoost Classifier")
    clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    with st.spinner("Training in progress..."):
        clf.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = clf.predict(X_test)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix with Plotly
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = px.imshow(cm, text_auto=True, labels={'x': "Predicted", 'y': "Actual"}, x=label_mapping.keys(), y=label_mapping.keys())
    cm_fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(cm_fig)

    # Enhanced Traffic Analysis and Insights
    st.subheader("Traffic Analysis and Insights")
    attack_count = (y == label_mapping.get('attack', -1)).sum()
    normal_count = (y == label_mapping.get('benign', -1)).sum()
    st.write(f"Total Attack Traffic: {attack_count}")
    st.write(f"Total Normal Traffic: {normal_count}")
    st.write(f"Attack to Normal Traffic Ratio: {attack_count / max(normal_count, 1):.2f}")

    # Analyze Packet Rate, Byte Rate, and Duration (if available)
    if {'pktcount', 'bytecount', 'tot_dur'}.issubset(data.columns):
        st.subheader("Detailed Feature Analysis")

        # Packet Rate Analysis
        pkt_rate_summary = filtered_data['pktcount'].describe()
        st.write("Packet Count Summary:")
        st.write(pkt_rate_summary)

        # Byte Rate Analysis
        byte_rate_summary = filtered_data['bytecount'].describe()
        st.write("Byte Count Summary:")
        st.write(byte_rate_summary)

        # Duration Analysis
        duration_summary = filtered_data['tot_dur'].describe()
        st.write("Total Duration Summary:")
        st.write(duration_summary)

        st.info("Recommendation: High packet or byte rates with low duration may indicate potential DDoS patterns.")

    # Identify common IPs in attack data if 'src' column exists
    if 'src' in data.columns:
        attack_ips = filtered_data[filtered_data['label'] == label_mapping.get('attack', -1)]
        common_ips = attack_ips['src'].value_counts().head(5)
        st.subheader("Top Suspicious IPs")
        st.write("The following IP addresses have the highest frequency in attack traffic and may need blocking:")
        st.write(common_ips)

        # Plot common IPs
        ip_chart = px.bar(common_ips, x=common_ips.index, y=common_ips.values, labels={'x': 'Source IP', 'y': 'Attack Count'})
        ip_chart.update_layout(title="Top Suspicious IPs")
        st.plotly_chart(ip_chart)

        if not common_ips.empty:
            st.info("Recommendation: Consider blocking or monitoring these IP addresses to prevent repeated attacks.")

    # Time-Series Visualization with Altair, check if 'dt' column is available
    if 'dt' in filtered_data.columns:
        filtered_data['dt'] = pd.to_datetime(filtered_data['dt'], errors='coerce')
        filtered_data = filtered_data.dropna(subset=['dt'])
        attack_data = filtered_data[filtered_data['label'] != label_mapping.get('benign', -1)]
        benign_data = filtered_data[filtered_data['label'] == label_mapping.get('benign', -1)]

        st.subheader("Time-Series Visualization of Attacks")
        attack_chart = alt.Chart(attack_data).mark_line(color='red').encode(
            x='dt:T', y='bytecount:Q', tooltip=['dt', 'bytecount']
        ).properties(title='Attack Data Over Time')
        
        benign_chart = alt.Chart(benign_data).mark_line(color='green').encode(
            x='dt:T', y='bytecount:Q', tooltip=['dt', 'bytecount']
        ).properties(title='Benign Data Over Time')
        
        st.altair_chart(attack_chart + benign_chart, use_container_width=True)

        # Real-Time DDoS Alert Notification
        latest_attack_time = attack_data['dt'].max() if not attack_data.empty else None
        st.subheader("Real-Time DDoS Attack Alerts")
        if latest_attack_time:
            time_since_last_attack = dt.datetime.now() - latest_attack_time
            if time_since_last_attack < dt.timedelta(minutes=5):
                st.error(f"ALERT: DDoS attack detected at {latest_attack_time}! Immediate action required.")
            else:
                st.success(f"No recent DDoS attacks detected. Last recorded attack was at {latest_attack_time}.")
        else:
            st.success("No DDoS attacks detected in the current dataset.")
    else:
        st.warning("Timestamp column is missing or improperly formatted. Time-series visualization and alerts are disabled.")
