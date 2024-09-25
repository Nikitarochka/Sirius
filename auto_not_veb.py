import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont
from rembg import remove
import requests
import uuid
import json

# Авторизация для GigaChat
client_id = ''
secret = ''
auth = ''

translator = GoogleTranslator(source='en', target='ru')

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def smooth_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

def remove_background(image, model_choice='rembg'):
    return remove(image)

def replace_background(image, background_color=(255, 255, 255)):
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    background_color = hex_to_rgb(background_color) if isinstance(background_color, str) else background_color
    texture = np.full(image.shape, background_color, dtype=np.uint8)
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    inverse_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(texture, texture, mask=inverse_mask)
    return cv2.add(foreground, background)

def get_token(auth_token, scope='GIGACHAT_API_PERS'):
    rq_uid = str(uuid.uuid4())
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json', 'RqUID': rq_uid, 'Authorization': f'Basic {auth_token}'}
    payload = {'scope': scope}
    
    response = requests.post(url, headers=headers, data=payload, verify=False)
    response.raise_for_status()
    return response.json()['access_token']

def get_chat_completion(auth_token, user_message):
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    payload = json.dumps({"model": "GigaChat", "messages": [{"role": "user", "content": user_message}], "temperature": 1, "top_p": 0.1, "n": 1, "stream": False, "max_tokens": 512, "repetition_penalty": 1, "update_interval": 0})
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'Authorization': f'Bearer {auth_token}'}

    response = requests.post(url, headers=headers, data=payload, verify=False)
    response.raise_for_status()
    return response.json()

def recognize_object(image, classes_content):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    labels = classes_content.splitlines()
    _, predicted_idx = torch.max(output, 1)
    object_name = labels[predicted_idx.item()]
    object_name = object_name.replace("_", " ")

    return translator.translate(object_name), predicted_idx

def add_text_above_image(image, text, output_path, font_path, background_color=(240, 240, 240), text_color=(0, 0, 0), font_size=35):
    image_width, image_height = image.size
    text_zone_height = 100
    new_image_height = image_height + text_zone_height
    new_image = Image.new("RGB", (image_width, new_image_height), background_color)
    new_image.paste(image, (0, text_zone_height))

    draw = ImageDraw.Draw(new_image)
    font = ImageFont.truetype(font_path, size=font_size)

    max_line_length = image_width - 100
    lines = []
    words = text.split()
    current_line = ""

    for word in words:
        if draw.textlength(current_line + " " + word, font=font) <= max_line_length:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    y_offset = (text_zone_height - len(lines) * (font.size + 10)) // 2
    for line in lines:
        line_width = draw.textlength(line, font=font)
        draw.text(((image_width - line_width) // 2, y_offset), line, font=font, fill=text_color)
        y_offset += font.size + 10

    new_image.save(output_path)

def process_images_in_folder(folder_path, output_folder, classes_file_path, font_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    classes_content = open(classes_file_path, 'r').read()

    token = get_token(auth)
    font_size = 8
    text_color = (0, 0, 0)  # Черный
    background_color = (128, 128, 128)  # Серый

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)

            image = Image.open(image_path)
            image = np.array(image)

            background_removed_image = remove_background(image)
            result_image = replace_background(background_removed_image, background_color)
            object_name, _ = recognize_object(image, classes_content)

            description = f"Создай рекламное описание товара: {object_name}. Примерно 15 слов."
            response = get_chat_completion(token, description)
            generated_description = response['choices'][0]['message']['content']

            add_text_above_image(Image.fromarray(result_image), generated_description, output_path, font_path, background_color=background_color, text_color=text_color, font_size=font_size)

    print("Обработка завершена.")

if __name__ == "__main__":
    folder_path = "C:\\Users\\Acer\\Documents\\Python\\SiriusML\\images\\sirius_data"
    output_folder = "C:\\Users\\Acer\\Documents\\Python\\SiriusML\\processed_images"
    classes_file_path = "C:\\Users\\Acer\\Documents\\Python\\SiriusML\\imagenet_classes.txt"
    font_path = "C:\\Users\\Acer\\Documents\\Python\\SiriusML\\Шрифты\\Montserrat-Medium.ttf"
    
    process_images_in_folder(folder_path, output_folder, classes_file_path, font_path)
