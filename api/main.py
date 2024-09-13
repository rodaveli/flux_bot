from flask import Flask, request, send_from_directory, url_for
from twilio.twiml.messaging_response import MessagingResponse
import os
import requests
import uuid
import threading
from twilio.rest import Client
import mimetypes

# Firebase imports
import firebase_admin
from firebase_admin import credentials, storage

app = Flask(__name__)

# Directory to save user images and generated images
USER_IMAGES_DIR = 'user_images'
GENERATED_IMAGES_DIR = 'static/generated'

os.makedirs(USER_IMAGES_DIR, exist_ok=True)
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# In-memory storage for user sessions (consider using a database in production)
user_sessions = {}

# Twilio credentials (use environment variables in production)
# Twilio credentials
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_NUMBER = os.environ.get('TWILIO_WHATSAPP_NUMBER')

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize Firebase Admin SDK
firebase_credentials = {
    "type": os.environ.get("FIREBASE_TYPE"),
    "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
    "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.environ.get("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
    "auth_uri": os.environ.get("FIREBASE_AUTH_URI"),
    "token_uri": os.environ.get("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.environ.get("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_X509_CERT_URL")
}

cred = credentials.Certificate(firebase_credentials)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'fal-whatsapp-thing.appspot.com'
})

@app.route('/whatsapp', methods=['POST'])
def whatsapp_bot():
    from_number = request.values.get('From')
    num_media = int(request.values.get('NumMedia', 0))
    body = request.values.get('Body', '').strip().lower()

    # Initialize response
    response = MessagingResponse()
    msg = response.message()

    # Start a new session or retrieve existing one
    session = user_sessions.get(from_number, {'state': 'awaiting_image_count', 'images': [], 'lora_model_path': None})

    if body == 'restart':
        session = {'state': 'awaiting_image_count', 'images': [], 'lora_model_path': None}
        user_sessions[from_number] = session
        msg.body("Session restarted. How many images will you provide for training the LORA model?")
        return str(response)

    if session['state'] == 'awaiting_image_count':
        if body.isdigit():
            session['expected_image_count'] = int(body)
            session['state'] = 'collecting_images'
            user_sessions[from_number] = session
            msg.body(f"Please send {session['expected_image_count']} images for training.")
        else:
            msg.body("Please enter the number of images you will provide for training.")
    elif session['state'] == 'collecting_images':
        if num_media > 0:
            # Process incoming images
            for i in range(num_media):
                media_url = request.values.get(f'MediaUrl{i}')
                media_content_type = request.values.get(f'MediaContentType{i}')
                # Download the image
                image_data = requests.get(media_url).content
                # Save the image
                image_filename = f"{uuid.uuid4()}{get_file_extension(media_content_type)}"
                image_path = os.path.join(USER_IMAGES_DIR, image_filename)
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                # Store the image path in the session
                session['images'].append(image_path)
            remaining = session['expected_image_count'] - len(session['images'])
            if remaining > 0:
                msg.body(f"Received image(s). Please send {remaining} more image(s).")
            else:
                msg.body("All images received. Training your LORA model now. This may take a few minutes.")
                session['state'] = 'training'
                user_sessions[from_number] = session
                # Start LORA training asynchronously
                threading.Thread(target=train_lora_model, args=(from_number,)).start()
        else:
            msg.body("Please send images for training.")
    elif session['state'] == 'training':
        msg.body("Your LORA model is still training. Please wait.")
    elif session['state'] == 'ready':
        if body:
            msg.body("Generating image based on your prompt. This may take a moment.")
            user_sessions[from_number] = session
            threading.Thread(target=generate_and_send_image, args=(from_number, body)).start()
        else:
            msg.body("Please send a prompt to generate an image.")
    else:
        msg.body("An error occurred. Please type 'restart' to start over.")

    return str(response)

def get_file_extension(mime_type):
    return mimetypes.guess_extension(mime_type) or ''

def train_lora_model(user_number):
    session = user_sessions[user_number]
    # Upload images and get a data URL
    images_data_url = upload_images_to_storage(session['images'])
    # Call Fal.ai API to train the LORA model
    import fal_client
    handler = fal_client.submit(
        "fal-ai/flux-lora-fast-training",
        arguments={
            "images_data_url": images_data_url,
            "create_masks": True,
            "iter_multiplier": 1
        },
    )
    result = handler.get()
    lora_model_url = result['lora_model_url']
    # Update session
    session['lora_model_path'] = lora_model_url
    session['state'] = 'ready'
    user_sessions[user_number] = session
    # Notify user
    send_whatsapp_message(user_number, "Your LORA model is ready! Please send me a prompt to generate an image.")

def generate_and_send_image(user_number, prompt):
    session = user_sessions[user_number]
    # Call Fal.ai API to generate image
    import fal_client
    handler = fal_client.submit(
        "fal-ai/flux-lora",
        arguments={
            "prompt": prompt,
            "image_size": "landscape_4_3",
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "num_images": 1,
            "enable_safety_checker": False,
            "output_format": "jpeg",
            "loras": [{
                "path": session['lora_model_path'],
                "scale": 1
            }],
            "sync_mode": True
        },
    )
    result = handler.get()
    # Save the generated image temporarily in /tmp directory
    image_data = result['images'][0]
    image_filename = f"{uuid.uuid4()}.jpeg"
    temp_image_path = os.path.join('/tmp', image_filename)
    with open(temp_image_path, 'wb') as f:
        f.write(image_data)

    # Upload generated image to Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(f'generated_images/{image_filename}')
    blob.upload_from_filename(temp_image_path)
    blob.make_public()

    # Get the public URL
    media_url = blob.public_url

    send_whatsapp_message(user_number, "Here is your generated image:", media_url)

def send_whatsapp_message(to_number, message, media_url=None):
    client.messages.create(
        body=message,
        from_=TWILIO_WHATSAPP_NUMBER,
        to=to_number,
        media_url=[media_url] if media_url else None
    )

def upload_images_to_storage(image_paths):
    # Create a ZIP archive of the images in /tmp directory
    zip_filename = f"{uuid.uuid4()}.zip"
    zip_filepath = os.path.join('/tmp', zip_filename)
    import zipfile
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        for image_path in image_paths:
            # Read image data from Firebase Storage
            bucket = storage.bucket()
            blob = bucket.blob(image_path)
            image_data = blob.download_as_bytes()
            # Write image data to zip
            zipf.writestr(os.path.basename(image_path), image_data)
    # Upload ZIP file to Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(f'user_data/{zip_filename}')
    blob.upload_from_filename(zip_filepath)
    # Make the blob publicly accessible
    blob.make_public()
    # Return the public URL
    image_data_url = blob.public_url
    return image_data_url

@app.route('/static/generated/<path:filename>')
def serve_generated_image(filename):
    return send_from_directory('static/generated', filename)
