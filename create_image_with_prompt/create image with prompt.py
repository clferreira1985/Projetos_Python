from flask import Flask, request, render_template, jsonify
import openai
from Key_OpenAI import API_KEY

app = Flask(__name__)

# Configure the OpenAI API key
client = openai.Client(api_key=API_KEY)

# Placeholder for valid tokens (in a real scenario, use a secure method to manage tokens)
VALID_TOKENS = {"YOUR_API_TOKEN"}

def create_image(prompt):
    response = client.images.generate(
        model='dall-e-2',
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response.data[0].url
    return image_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image-generation', methods=['POST'])
def generate_image():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Unauthorized"}), 401

    token = auth_header.split(" ")[1]
    if token not in VALID_TOKENS:
        return jsonify({"error": "Unauthorized"}), 401

    if request.is_json:
        data = request.get_json()
        if 'prompt' not in data:
            return jsonify({"error": "No prompt provided"}), 400
        
        prompt = data['prompt']
        try:
            image_url = create_image(prompt)
            return jsonify({"prompt": prompt, "image_url": image_url})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Request must be JSON"}), 415

if __name__ == '__main__':
    app.run(debug=True)
