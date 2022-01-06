import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request
from service import detect

# flask web service
app = Flask(__name__, template_folder="web")


@app.route('/detect/imageDetect', methods=['post'])
def predict():
    # step 1. receive image
    file = request.form.get('imageBase64Code')
    image = Image.open(BytesIO(base64.b64decode(file)))

    # step 2. detect image
    image_array = detect(image)

    # step 3. convert image_array to byte_array
    img = Image.fromarray(image_array)
    img_byte_array = BytesIO()
    img.save(img_byte_array, format='JPEG')

    # step 4. return image_info to page
    image_info = base64.b64encode(img_byte_array.getvalue()).decode('ascii')
    return image_info


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, port=8081)
