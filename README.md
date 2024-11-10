# Python-System

# Setting Up and Running a Streamlit Environment

## Step-by-Step Guide

### Step 1: Create a Virtual Environment
To create a virtual environment for your Streamlit project, run the following command:
```bash
python3 -m venv ~/downloads/streamlit-env
```

### Step 2: Activate the Virtual Environment
Activate your virtual environment using the command below:
```bash
source ~/downloads/streamlit-env/bin/activate
```

### Step 3: Install Streamlit in the Virtual Environment
Install Streamlit by running:
```bash
pip install streamlit
```

### Step 4: Install All Required Packages
Download and install all necessary packages for the project:
```bash
pip install altair==4.2.2 attrs==24.2.0 beautifulsoup4==4.12.3 blinker==1.9.0 cachetools==5.5.0 certifi==2024.8.30 charset-normalizer==3.4.0 click==8.1.7 contourpy==1.3.0 cycler==0.12.1 entrypoints==0.4 et_xmlfile==2.0.0 fonttools==4.54.1 gitdb==4.0.11 GitPython==3.1.43 idna==3.10 Jinja2==3.1.4 joblib==1.4.2 jsonschema==4.23.0 jsonschema-specifications==2024.10.1 kiwisolver==1.4.7 markdown-it-py==3.0.0 MarkupSafe==3.0.2 matplotlib==3.9.2 mdurl==0.1.2 narwhals==1.13.3 numpy==2.1.3 openpyxl==3.1.5 packaging==24.2 pandas==2.2.3 pillow==11.0.0 pip==24.2 protobuf==5.28.3 pyarrow==18.0.0 pydeck==0.9.1 Pygments==2.18.0 pyparsing==3.2.0 python-dateutil==2.9.0.post0 pytz==2024.2 referencing==0.35.1 requests==2.32.3 rich==13.9.4 rpds-py==0.21.0 scikit-learn==1.5.2 scipy==1.14.1 seaborn==0.13.2 six==1.16.0 smmap==5.0.1 soupsieve==2.6 streamlit==1.40.0 tenacity==9.0.0 threadpoolctl==3.5.0 toml==0.10.2 toolz==1.0.0 tornado==6.4.1 typing_extensions==4.12.2 tzdata==2024.2 urllib3==2.2.3 xgboost==2.1.2
```

### Step 5: Run Your Streamlit App
Finally, run your Streamlit app by executing:
```bash
streamlit run Enhanced.py
```

## Notes
- Ensure your virtual environment is activated before installing packages or running your Streamlit app.
- Replace `Enhanced.py` with your own script name if different.

Enjoy building and running your Streamlit applications!




# About Enhanced.py

## Overview
`Enhanced.py` is a Streamlit-based application designed to simulate network flow, provide real-time DDoS attack detection, and include a website vulnerability scanner. The app leverages a range of Python libraries for data analysis, machine learning, and web scraping to deliver an interactive and comprehensive user experience.

## Key Features
- **Streamlit UI**:
  - Interactive and user-friendly interface for uploading datasets in `.xlsx` format.
  - Sidebar toggle to enable or disable "Protection Mode," displaying real-time feedback on status.

- **Data Handling**:
  - Utilizes `pandas` for data manipulation and displaying uploaded data within the app.
  - Supports user-uploaded datasets for custom analysis.

- **Machine Learning Integration**:
  - Uses `XGBClassifier` from XGBoost for training and DDoS attack prediction.
  - Implements `scikit-learn` for data preprocessing (`train_test_split`, `StandardScaler`, `LabelEncoder`) and evaluation (`classification_report`, `confusion_matrix`, `accuracy_score`).

- **Visualization Tools**:
  - Incorporates `matplotlib` and `seaborn` for visualizing data insights and model performance.
  - Displays real-time alerts and notifications using Streamlit components.

- **Web Scraping**:
  - Includes `BeautifulSoup` and `requests` for basic website vulnerability scanning by scraping and analyzing webpage content.

- **Real-Time Feedback**:
  - Provides notifications for actions such as enabling protection mode and running the detection model.
  - Uses timers and progress indicators to keep users informed of ongoing processes.

## Functional Steps
1. **Import Libraries**:
   - Essential packages for data processing, visualization, web scraping, and machine learning.

2. **Upload Functionality**:
   - A Streamlit-based file uploader allowing users to input their `.xlsx` datasets for analysis.

3. **Protection Mode Toggle**:
   - Sidebar checkbox to activate or deactivate "Protection Mode," displaying success or warning notifications accordingly.

4. **Data Processing and Machine Learning**:
   - Processes the uploaded dataset, applies machine learning algorithms, and outputs predictions and model metrics.

5. **Website Vulnerability Scanning**:
   - Optional feature to conduct a basic vulnerability scan by analyzing a user-specified URL using web scraping.

## Libraries Used
- `pandas`
- `streamlit`
- `matplotlib`
- `seaborn`
- `requests`
- `BeautifulSoup`
- `scikit-learn`
- `xgboost`
- `numpy`
- `time`, `datetime`, `random`

## How to Use
1. Run the app:
   ```bash
   streamlit run Enhanced.py
   ```
2. Upload your dataset using the provided interface.
3. Toggle "Protection Mode" as needed.
4. Review analysis outputs, visualizations, and DDoS detection results.

## Notes
- Ensure all required libraries are installed before running the app.
- Replace `Enhanced.py` with the name of your script if different.

Enjoy using this powerful tool for network analysis and cybersecurity simulations!



