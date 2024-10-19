from flask import Flask, render_template, request
from pathlib import Path
from pymongo import MongoClient
import google.generativeai as genai
import os
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

client = MongoClient("mongodb://localhost:27017/")
db = client['Feature_Extraction']
collection = db['Image_Features']

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure the Google Gemini API
genai.configure(api_key="AIzaSyAlKyF3PpDiNc4KUAr7uKDA-v7HtDQcpDA")

def image_format(image_path):
    img = Path(image_path)

    if not img.exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    if img.suffix in ['.jpeg', '.jpg']:
        mime_type = "image/jpeg"
    elif img.suffix == '.png':
        mime_type = "image/png"
    else:
        raise ValueError("Unsupported image format. Only JPEG and PNG are allowed.")

    image_parts = [
        {
            "mime_type": mime_type,
            "data": img.read_bytes()
        }
    ]
    return image_parts


def gemini_output(image_path, system_prompt, user_prompt):
    image_info = image_format(image_path)
    input_prompt = [system_prompt, image_info[0], user_prompt]

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(input_prompt)

    response_text = response.text

    # Format response for HTML rendering
    formatted_response = response_text.replace("##", "<h6><br>").replace("* ", "<li>").replace(" - ", "</li><li>")
    formatted_response = formatted_response.replace("**", "</h6>").replace("<li>", "<li><p>")
    formatted_response = "<ul>" + formatted_response + "</ul>"  # Wrap the response in a list

    return formatted_response

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Define the system instruction separately
system_instruction = (
    "You are an expert Feature Extractor and Image Analyzer. "
    "You should only give details as a List with a single Description. "
    "You possess the ability to extract colors, species, gender, places, "
    "daytime, objects, patterns, etc. Also, details must be bolded or highlighted "
    "with side headings. Side headings are required! "
    "You should not respond if you can't read the file and "
    "should say 'couldn't extract details'. "
    "Output should be in a list format."
)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part'

        file = request.files['image']

        if file.filename == '':
            return 'No selected file'

        if file:
            # Save the file locally
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print(f"Saved file to {file_path}")

            # Generate the response using the model
            user_prompt = "Extract all relevant information from the image."
            output = gemini_output(file_path, system_prompt=system_instruction, user_prompt=user_prompt)
            history_data = {
                'image_name': file.filename,
                'image_path': file_path,
                'response': output,
                'timestamp': datetime.now(),
                'gemini_history': [
                    {
                        "role": "user",
                        "image": file.filename,
                    },
                    {
                        "role": "model",
                        "response": output,  # Dynamic response
                    }
                ]
            }

            # Insert the data into MongoDB and check the result
            try:
                result = collection.insert_one(history_data)
                print(f"Inserted document with ID: {result.inserted_id}")
            except Exception as e:
                print(f"An error occurred: {e}")
                return "Error inserting data into MongoDB"

            return render_template('result.html', response=output)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change to a different port
