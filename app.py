import numpy as np
import joblib
import gradio as gr

#load the dataset  
model = joblib.load("automobile (1).pkl")
scaler = joblib.load("scaler (1).pkl")

# Define cluster labels
cluster_names = {
    0: "Economy Cars",
    1: "Luxury Cars",
    2: "High Performance Cars"
}

#prediction function
def predict_sp(
        year,
        engine_size,
        mileage,
        horespower,
        owners,
        torque,
        accident_history,
        fuel_efficiency
):
    input_data = np.array([[
            year,
            engine_size,
            mileage,
            horespower,
            owners,
            torque,      
            accident_history,       
            fuel_efficiency   
        ]])
    input_scaled = scaler.transform(input_data)
    
    cluster = model.predict(input_data)[0]

    # Return the corresponding category
    return f"Predicted Category: {cluster_names.get(cluster, 'Unknown Category')}"

#Gradio Interface
app_ = gr.Interface(
    fn = predict_sp,

    inputs =   [
        gr.Number(label="Year"),
        gr.Number(label="Engine Size (cc)"),
        gr.Number(label="Mileage (km)"),
        gr.Number(label="Horsepower"),
        gr.Number(label="Number of Owners"),
        gr.Number(label="Torque (Nm)"),
        gr.Number(label="Accident History(0 = No, 1 = Yes)"),
        gr.Number(label="Fuel Efficiency (km/l)"),
    ],
    outputs=gr.Textbox(label="Predicted Car Category"),
    title="Automobile Classification using K-Means Clustering",
    description="Enter the vehicle details to determine which cluster (car category) it belongs to."
)

app_.launch(
    server_name = "0.0.0.0",
    server_port = 7860
)