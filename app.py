import joblib
import gradio as gr
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Define prediction function
def predict_price(area, bedrooms, bathrooms):
    features = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(features)[0]
    return f"${prediction:,.2f}"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Area (sq ft)"),
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms")
    ],
    outputs="text",
    title="Housing Price Predictor",
    description="Enter the home details to get an estimated price."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
