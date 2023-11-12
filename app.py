import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from flask import Flask, request, send_file
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFilter, ImageOps,ImageFont
import textwrap
from io import BytesIO
from webcolors import rgb_to_name, hex_to_rgb, CSS3_HEX_TO_NAMES
from flask import Flask, request, send_file
from PIL import Image
from diffusers import AutoPipelineForImage2Image
import os
import torch
from webcolors import rgb_to_name, hex_to_rgb, CSS3_HEX_TO_NAMES
from io import BytesIO
app = Flask(__name__, static_folder='static')

# stable-diffusion model pipline
 

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def rgb_to_color_name(rgb):
    try:
        color_name = rgb_to_name(rgb)
        return color_name
    except ValueError:
        closest_match = min(
            CSS3_HEX_TO_NAMES.items(),
            key=lambda item: sum((a - b) ** 2 for a, b in zip(hex_to_rgb(item[0]), rgb)))[1]
        return closest_match



def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))



def rgb_to_rgba(rgb, alpha):
    rgb = [max(0, min(255, val)) for val in rgb]
    rgba = tuple(rgb + [alpha])
    return rgba

def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    rgba = rgb + (255,)
    
    return rgba

def process_image(input_path,input_path_logo,text_input,button_text,hex_code,check_ai):
        input_image_path = input_path
        hex_color_code = hex_code
        logo_path = input_path_logo
        text=text_input
        button_text=button_text
        corner_radius = 20
        path_ttf="timr45w.ttf"
        size_val=300
        #input format is changed by check_ai,check_ai==True if stable-diffusion model is used
        if(check_ai==True):
          img = input_image_path.convert("RGBA")
        else:
          img = Image.open(input_image_path).convert("RGBA")
        
        rgb_values = hex_to_rgb(hex_color_code)
        color_name = rgb_to_color_name(rgb_values)
        text_color=color_name+" color based"
        text0="ok "+text_color

        #changing shape of image 
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), img.size], corner_radius, fill=255)
        image_result_original = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 0)), img)
        image_result_original.putalpha(mask)
        ratio=image_result_original.width/image_result_original.height

        #adding background
        image_result=image_result_original.resize((int(size_val*ratio),size_val), Image.LANCZOS)
        background_size=(int(size_val*2.5),int(size_val*2.5))
        new_image = Image.new("RGB", background_size, "white") 
        width_bg, height_bg = new_image.size
        width_overlay, height_overlay = image_result.size
        x_position = (width_bg - width_overlay) // 2
        y_position = int(size_val*0.55)+20
        combined = Image.new("RGBA", new_image.size)
        combined.paste(new_image, (0, 0))
        combined.paste(image_result, (x_position, y_position), image_result)

        #adding logo
        logo_first = Image.open(logo_path).convert("RGBA")
        ratio_logo=logo_first.width/logo_first.height
        size_logo_val=int(size_val/2)
        logo=logo_first.resize((int(size_logo_val*ratio_logo),size_logo_val), Image.BICUBIC)
        width_overlay_logo, height_overlay_logo = logo.size
        x_position = (width_bg - width_overlay_logo) // 2
        y_position = 20

        combined_logo = Image.new("RGBA", combined.size)
        combined_logo.paste(combined, (0, 0))
        combined_logo.paste(logo, (x_position, y_position), logo)


        #adding top bar and bottom bar 
        rectangle_width = int(size_val*2.25)
        rectangle_height = 10
        radius = 50
        x_position = (combined_logo.width - rectangle_width) // 2
        y_position =int((-1400/size_val)) 
        rgb_code=hex_to_rgb(hex_color_code)
        rgba_values_0 = rgb_to_rgba(rgb_code,0)
        rgba_values_255 = rgb_to_rgba(hex_to_rgb(hex_color_code),255)
        rounded_rectangle = Image.new('RGBA', (rectangle_width, rectangle_height), rgba_values_0)
        draw = ImageDraw.Draw(rounded_rectangle)
        draw.rounded_rectangle([(0, 0), (rectangle_width, rectangle_height)], radius, fill=hex_to_rgba(hex_color_code))
        combined_logo.paste(rounded_rectangle, (x_position, y_position), rounded_rectangle)

        y_position_bottom = int(2.48*size_val)
        rounded_rectangle_bottom = Image.new('RGBA', (rectangle_width, rectangle_height), rgb_to_rgba(hex_to_rgb(hex_color_code),0))
        draw = ImageDraw.Draw(rounded_rectangle)
        draw.rounded_rectangle([(0, 0), (rectangle_width, rectangle_height)], radius, fill=hex_to_rgba(hex_color_code))
        combined_logo.paste(rounded_rectangle, (x_position, y_position_bottom), rounded_rectangle)

        #adding frame
        border_size = 2
        border_color = (0, 0, 0) 
        image_with_frame = ImageOps.expand(combined_logo, border=border_size, fill=border_color)
        image_width, image_height = image_with_frame.size
        draw = ImageDraw.Draw(image_with_frame)

        #adding punchline text
        font_size = 45
        font = ImageFont.truetype(path_ttf, font_size)
        max_text_width = image_width - 40 
        wrapped_text = textwrap.fill(text, width=30) 
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = ((image_width - text_width) // 2, (image_height - text_height) // 2 + image_height // 4)
        text_color = rgb_code 
        draw.multiline_text(text_position, wrapped_text, font=font, fill=text_color, align="center")
        #adding button
        button_color = rgb_code 
        button_font_size = 30
        button_font = ImageFont.truetype(path_ttf, button_font_size)
        button_width = 300
        button_height = 45
        button_position = ((image_width - button_width) // 2, text_position[1] + text_height + 30)  # Yüksekliği ayarlandı

        draw.rounded_rectangle([button_position, (button_position[0] + button_width, button_position[1] + button_height)],
                            fill=button_color, outline=None, width=0, radius=10)

        button_text_position = ((image_width - button_width) // 2 + (button_width - draw.textlength(button_text, font=button_font)) // 2,
                            button_position[1] + 5)  
        button_text_color = (255, 255, 255)  


        draw.text(button_text_position, button_text, font=button_font, fill=button_text_color)

  
        image_with_frame

        output_buffer = BytesIO()

      
        image_with_frame.save(output_buffer, format="PNG")

       
        output_buffer.seek(0)

        return output_buffer
#banner without stable-diffusion model image
@app.route('/get-without-diff', methods=['POST'])
def upload_file_without_diff():
    if 'file' not in request.files or 'file_logo' not in request.files:
        return 'Both files are required', 400

    file = request.files['file']
    file_logo = request.files['file_logo']
    hex_code = request.form.get('hex_input')  
    text_input = request.form.get('text_input')  
    button_text = request.form.get('button_text_input')  

    if file.filename == '' or file_logo.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename) and file_logo and allowed_file(file_logo.filename):
        
        output_buffer = process_image(file, file_logo, text_input,button_text, hex_code,False)

       
        return send_file(output_buffer, as_attachment=True, download_name='banner.png', mimetype='image/png'), 200
    else:
        return 'Invalid file type. Please enter files of type png, jpg, jpeg, gif', 400
    
 # stable-diffusion model image
@app.route('/get-diff', methods=['POST'])
def get_image():
    
    uploaded_file = request.files['file_diff']
    hex_color_code = request.form.get('hex_diff')
    rgb_values = hex_to_rgb(hex_color_code)
    color_name = rgb_to_color_name(rgb_values)
    text_color = color_name + " color based"
    #prompt text with hex color input
    prompt_text = request.form.get('prompt') + " " + text_color

    init_image = Image.open(uploaded_file).convert("RGB")

    prompt = prompt_text

    #generating image
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    processed_image = pipe(prompt, image=init_image, num_inference_steps=10, image_guidance_scale=1).images

    
    img_byte_array = BytesIO()
    processed_image[0].save(img_byte_array, format='PNG')
    img_byte_array.seek(0)
    
    return send_file(img_byte_array, mimetype='image/png')

 #banner with stable-diffusion model image- 
@app.route('/get-result', methods=['POST'])
def upload_file():
    if 'file_diff' not in request.files or 'file_logo' not in request.files:
        return 'Both files are required', 400
    uploaded_file = request.files['file_diff']
    hex_color_code = request.form.get('hex_diff')
    rgb_values = hex_to_rgb(hex_color_code)
    color_name = rgb_to_color_name(rgb_values)
    text_color = color_name + " color based"
    file_logo = request.files['file_logo']
    hex_code = request.form.get('hex_input')  
    text_input = request.form.get('text_input')  
    button_text = request.form.get('button_text_input')  

    prompt_text = request.form.get('prompt') + " " + text_color

  
    init_image = Image.open(uploaded_file).convert("RGB")

    prompt = prompt_text

    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    processed_image = pipe(prompt, image=init_image, num_inference_steps=10, image_guidance_scale=1).images

    if uploaded_file.filename == '' or file_logo.filename == '':
        return 'No selected file', 400

    if processed_image[0] and allowed_file(uploaded_file.filename) and file_logo and allowed_file(file_logo.filename):
        
        output_buffer = process_image(processed_image[0], file_logo, text_input,button_text, hex_code,True)

       
        return send_file(output_buffer, as_attachment=True, download_name='banner.png', mimetype='image/png'), 200
    else:
        return 'Invalid file type. Please enter files of type png, jpg, jpeg, gif', 400

@app.route('/', methods=['GET'])
def get_app():
    return "App works."
if __name__ == '__main__':
    app.run(debug=True)


