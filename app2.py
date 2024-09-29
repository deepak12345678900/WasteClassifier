from flask import Flask, request, render_template
from PIL import Image
import io
import inference  # Assuming your custom model inference module

app = Flask(__name__)

# Initialize the model
model = inference.get_model("waste_classifer/1", api_key="ba3svItNcD46dMWVcJKJ")

@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize variables for predictions
    prediction = None
    glass_conf = None
    battery_conf = None
    biological_conf = None
    metal_conf = None
    paper_conf = None
    plastic_conf = None

    if request.method == 'POST':
        # Check if the clear output button was clicked
        if 'clear_output' in request.form:
            # Reset all prediction-related variables
            return render_template('index.html')

        # Check if a file was uploaded
        if 'file' not in request.files or request.files['file'].filename == '':
            # No need to show any error; just return the empty template
            return render_template('index.html')

        uploaded_image = request.files['file']
        if uploaded_image:
            img = Image.open(io.BytesIO(uploaded_image.read()))
            pred = model.infer(image=img)
            
            # Extract prediction results
            prediction = pred[0].predicted_classes[0].upper()
            glass_conf = round(pred[0].predictions['Glass'].confidence, 3)
            battery_conf = round(pred[0].predictions['battery'].confidence, 3)
            biological_conf = round(pred[0].predictions['biological'].confidence, 3)
            metal_conf = round(pred[0].predictions['metal'].confidence, 3)
            paper_conf = round(pred[0].predictions['paper'].confidence, 3)
            plastic_conf = round(pred[0].predictions['plastic'].confidence, 3)

    return render_template('index.html', 
                           prediction=prediction,
                           glass_conf=glass_conf,
                           battery_conf=battery_conf,
                           biological_conf=biological_conf,
                           metal_conf=metal_conf,
                           paper_conf=paper_conf,
                           plastic_conf=plastic_conf)

if __name__ == '__main__':
    app.run(debug=True)
