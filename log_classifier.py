import streamlit as st
import pandas as pd
import pickle
from io import StringIO
import re
import base64

# Function to parse the log file
def parse_log_file(log_file_content):
    logs = log_file_content.splitlines()
    data = []
    log_entry = ""
    timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+\d{2}:\d{2}$')

    for line in logs:
        if timestamp_pattern.match(line.strip()):
            if log_entry:
                data.append(log_entry.strip())
            log_entry = line.strip() + " "
        else:
            log_entry += line.strip() + " "
    
    if log_entry:
        data.append(log_entry.strip())

    return pd.DataFrame(data, columns=['log_message'])


# Function to classify logs and count severities
def classify_logs(log_file_content):
    try:
        # Load the trained model and TF-IDF vectorizer
        model = pickle.load(open('model/log_classifier.pkl', 'rb'))
        vectorizer = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None

    # Parse the log file
    df_logs = parse_log_file(log_file_content)

    if df_logs is None or df_logs.empty:
        st.error("No logs found to classify.")
        return None

    try:
        # Transform log messages to TF-IDF features
        log_messages_tfidf = vectorizer.transform(df_logs['log_message'])

        # Predict the severity using the model
        df_logs['predicted_severity'] = model.predict(log_messages_tfidf)
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

    # Map severity numbers to labels
    severity_mapping = {0: 'Information', 1: 'Warning', 2: 'Error', 3: 'Critical'}
    df_logs['predicted_severity_label'] = df_logs['predicted_severity'].map(severity_mapping)

    # Count the number of each severity
    severity_counts = df_logs['predicted_severity_label'].value_counts().to_dict()

    return df_logs, severity_counts

# Streamlit UI
st.title("Log File Severity Classifier")

uploaded_file = st.file_uploader("Choose a log file", type=["log", "txt"])

if uploaded_file is not None:
    # Read the file as string
    log_file_content = uploaded_file.read().decode("utf-8")
    
    # Display log file content for debugging (remove this in production)
    st.text_area("Log File Content", log_file_content, height=300)  # Display log content for debugging

    if st.button("Classify Logs"):
        classified_logs, severity_counts = classify_logs(log_file_content)
        
        if classified_logs is not None:
            st.success("Logs classified successfully!")

            # Display the counts of each severity with labels
            st.subheader("Severity Counts")
            st.write(severity_counts)
            
            # Convert the classified logs to CSV for download
            csv = classified_logs.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # Some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}" download="classified_logs.csv">Download Classified Logs</a>'
            st.markdown(href, unsafe_allow_html=True)
