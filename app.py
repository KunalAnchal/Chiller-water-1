from flask import Flask, render_template, request, send_file, jsonify
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import WriteOptions
import threading
import datetime
import time

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Define the local path to the dataset
DATASET_PATH = 'chillerdata.csv'

# InfluxDB 2.x Configuration
INFLUXDB_TOKEN = "Uxh3_M9yNWhCE-Ne9xeMYV_I0-sGXBBF3KELMTLDT8UUmcT__jUxAPVmzKmF-DC58dJvsBFovQwtYxrT5hOWeg=="
INFLUXDB_ORG = "IIIOT-INFOTECH"
INFLUXDB_BUCKET = "Machine Learning"
INFLUXDB_URL = "http://192.168.1.130:8086/"

# Initialize InfluxDB client with optimized write options
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=WriteOptions(
    batch_size=1000,
    flush_interval=10_000,
    jitter_interval=2_000,
    retry_interval=5_000
))

@app.route('/')
def index():
    return render_template('index.html', graph=None, data_table=None, excel_file=None, error=None)

@app.route('/predict', methods=['POST'])
def predict_and_plot():
    try:
        # Read the CSV data from the local path
        df = pd.read_csv(DATASET_PATH)

        # Ensure that the CSV file has the expected columns
        required_columns = ['Outside Temperature (F)', 'Dew Point (F)', 'Humidity (%)', 'Wind Speed (mph)',
                            'Pressure (in)', 'Month', 'Day', 'Hour']
        if not set(required_columns).issubset(df.columns):
            return render_template('index.html', error="The CSV file is missing required columns.", graph=None,
                                   data_table=None, excel_file=None)

        # Select only the required columns for prediction
        X = df[required_columns]

        # Perform predictions using the model
        df['Prediction'] = model.predict(X)

        # Create distribution plots for Global_active_power and Prediction
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.figure()

        # Combined distribution plot with custom colors
        sns.histplot(df['Chilled Water Rate (L/sec)'], label='Chilled Water Rate (L/sec)', color='yellow', kde=True)  # Yellow for actual
        sns.histplot(df['Prediction'], label='Prediction', color='red', kde=True)  # Red for predicted
        plt.title('Distribution of Chilled Water Rate (L/sec) and Prediction', color='white')
        plt.xlabel('Values', color='white')
        plt.ylabel('Density', color='white')
        plt.legend()

        # Customize for dark mode
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.gca().set_facecolor('black')
        plt.xticks(color='white')
        plt.yticks(color='white')  # Optional: Change y-tick labels color for contrast

        # Save the plot to a buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', facecolor='black')  # Ensure black background
        img_buffer.seek(0)
        graph = base64.b64encode(img_buffer.read()).decode('utf-8')

        # Generate data table
        data_table = df.to_html(classes='table table-condensed table-bordered table-striped')

        # Save the predicted data to an Excel file
        excel_file_path = 'predicted_data.xlsx'
        df.to_excel(excel_file_path, index=False)

        # Start a new thread to send data to InfluxDB
        threading.Thread(target=send_to_influxdb_continuously, args=(df, 0.1)).start()

        return render_template('index.html', graph=graph, data_table=data_table, excel_file=excel_file_path, error=None)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {e}", graph=None, data_table=None, excel_file=None)

def send_to_influxdb_continuously(df, delay=0.1):
    try:
        base_time = datetime.datetime.utcnow()

        for index, row in df.iterrows():
            try:
                timestamp = (base_time + datetime.timedelta(seconds=index * delay)).isoformat()

                point = Point("chilled_water_data") \
                    .field("Actual_chilled_water_rate", float(row['Chilled Water Rate (L/sec)'])) \
                    .field("Predicted_chilled_water_rate", float(row['Prediction'])) \
                    .time(timestamp)  # Assign a unique timestamp

                write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
                print(f"Row {index} written to InfluxDB at {timestamp}")

                time.sleep(delay)

            except Exception as e:
                print(f"Error writing row {index} to InfluxDB: {e}")

        print("All data has been sent to InfluxDB.")

    except Exception as e:
        print(f"Error in send_to_influxdb_continuously: {e}")

@app.route('/download_excel')
def download_excel():
    try:
        excel_file_path = request.args.get('excel_file')
        return send_file(excel_file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8057, debug=True)
