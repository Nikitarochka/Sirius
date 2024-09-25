import cv2
import numpy as np
import torch
from torchvision import models, transforms
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont
from rembg import remove
import streamlit as st
from torch.nn.functional import softmax
import requests
import uuid
import json

# Авторизация для GigaChat
client_id = ''
secret = ''
auth = ''

translator = GoogleTranslator(source='en', target='ru')

def hex_to_rgb(hex_color):
    """Преобразует цвет в формате HEX (#FFFFFF) в RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def smooth_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

def remove_background(image, model_choice='rembg'):
    if model_choice == 'rembg':
        result = remove(image)
    elif model_choice == 'deeplabv3':
        model = models.segmentation.deeplabv3_resnet101(
            weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).eval()
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)['out'][0]

        mask = output.argmax(0).byte().cpu().numpy()
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask = (mask * 255).astype(np.uint8)
        mask = smooth_mask(mask)

        result = cv2.bitwise_and(image, image, mask=mask)
    else:
        raise ValueError("Unknown model_choice, use 'rembg' or 'deeplabv3'")
    
    return result

def replace_background(image, background_color=(255, 255, 255)):
    """Заменяет фон изображения на указанный цвет."""
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    # Преобразование HEX-цвета в RGB
    background_color = hex_to_rgb(background_color) if isinstance(background_color, str) else background_color
    texture = np.full(image.shape, background_color, dtype=np.uint8)

    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    foreground = cv2.bitwise_and(image, image, mask=mask)
    inverse_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(texture, texture, mask=inverse_mask)

    final_image = cv2.add(foreground, background)
    final_image = cv2.normalize(final_image, None, 0, 255, cv2.NORM_MINMAX)

    return final_image

def get_token(auth_token, scope='GIGACHAT_API_PERS'):
    rq_uid = str(uuid.uuid4())
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': rq_uid,
        'Authorization': f'Basic {auth_token}'
    }
    payload = {'scope': scope}
    
    try:
        response = requests.post(url, headers=headers, data=payload, verify=False)
        response.raise_for_status()
        return response.json()['access_token']
    except requests.RequestException as e:
        print(f"Ошибка: {str(e)}")
        return None

def get_chat_completion(auth_token, user_message):
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    
    payload = json.dumps({
        "model": "GigaChat",
        "messages": [{"role": "user", "content": user_message}],
        "temperature": 1,
        "top_p": 0.1,
        "n": 1,
        "stream": False,
        "max_tokens": 512,
        "repetition_penalty": 1,
        "update_interval": 0
    })

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {auth_token}'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, verify=False)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Ошибка: {str(e)}")
        return None

def recognize_object(image, classes_content):
    """Распознает объект на изображении с помощью модели ResNet50."""
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

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

    if "_" in object_name:
        object_name = object_name.replace("_", " ")

    object_name_ru = translator.translate(object_name)
    return object_name_ru, predicted_idx 

# Измененная функция добавления текста над изображением
def add_text_above_image(image, text, output_path, font_path=None, background_color=(240, 240, 240), text_color=(0, 0, 0), font_size=35):
    image_width, image_height = image.size

    # Размеры зоны для текста сверху
    text_zone_height = 100

    # Создаем новое изображение с зоной для текста
    new_image_height = image_height + text_zone_height
    new_image = Image.new("RGB", (image_width, new_image_height), background_color)
    new_image.paste(image, (0, text_zone_height))

    draw = ImageDraw.Draw(new_image)

    # Устанавливаем путь к шрифту, если он предоставлен
    if font_path is not None:
        try:
            font = ImageFont.truetype(font_path, size=font_size)
        except IOError:
            st.error("Ошибка загрузки шрифта. Используется шрифт по умолчанию.")
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default() 

    max_line_length = image_width - 100 

    # Разделение текста на строки для подгонки по ширине
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

    # Проверка, что текст помещается в зону, и увеличение зоны, если необходимо
    while len(lines) * (font.size + 10) > text_zone_height:
        text_zone_height += 50  
        new_image = Image.new("RGB", (image_width, image_height + text_zone_height), background_color)
        new_image.paste(image, (0, text_zone_height))
        draw = ImageDraw.Draw(new_image)

    # Рисуем текст по центру зоны для текста
    y_offset = (text_zone_height - len(lines) * (font.size + 10)) // 2
    for line in lines:
        _, _, _, text_height = draw.textbbox((0, 0), line, font=font)
        line_width = draw.textlength(line, font=font)
        draw.text(((image_width - line_width) // 2, y_offset), line, font=font, fill=text_color)
        y_offset += text_height + 10

    # Сохраняем изображение
    new_image.save(output_path)


def main():
    st.title("Обработка изображений товаров")

    uploaded_file = st.file_uploader("Загрузите изображение товара", type=["jpg", "png", "jpeg"])
    classes_file = st.file_uploader("Загрузите файл с классами", type=["txt"])
    font_file = st.file_uploader("Загрузите шрифт (необязательно)", type=["ttf"])
    
    font_size = st.slider("Выберите размер шрифта", min_value=1, max_value=100, value=35)
    text_color = st.color_picker("Выберите цвет текста", "#000000") 

    background_color = st.color_picker("Выберите цвет фона", "#FFFFFF")
    background_file = st.file_uploader("Загрузите фон (необязательно)", type=["jpg", "png", "jpeg"])

    if background_file:
        background_image = Image.open(background_file)
        background_image = np.array(background_image)
    else:
        background_image = None  # Использовать цвет фона по умолчанию

    # Выбор стиля описания
    description_style = st.selectbox("Выберите стиль описания", ["Краткое", "Техническое", "Подробное", "Рекламное", "Креативное", "Маркетинговое"])
    # Добавить выбор модели в интерфейсе Streamlit
    model_choice = st.selectbox("Выберите модель для удаления фона", ["rembg", "deeplabv3"])
  
    if uploaded_file and classes_file:
        image = Image.open(uploaded_file)
        image = np.array(image)

        classes_content = classes_file.read().decode("utf-8")

        # Передавать выбранную модель в функцию remove_background
        with st.spinner("Удаление фона..."):
            background_removed_image = remove_background(image, model_choice=model_choice)

        if st.checkbox("Показать изображение без фона"):
            st.image(background_removed_image, caption="Изображение без фона", use_column_width=True)

        if st.button("Заменить фон"):
            result_image = replace_background(background_removed_image, background_image if background_image is not None else background_color)
            st.session_state['result_image'] = result_image
            st.image(result_image, caption="Изображение с новым фоном", use_column_width=True)

        object_name, predicted_idx = recognize_object(image, classes_content)
        st.write(f"Объект распознан: {object_name}")
        
        # Получаем текущее название объекта из состояния сессии или используем распознанное
        object_name = st.session_state.get('object_name', object_name)
        
        new_object_name = st.text_input("Введите новое название объекта (если нужно)", object_name)

        # Обновляем object_name при вводе нового названия
        if st.button("Заменить объект"):
            st.session_state['object_name'] = new_object_name.strip()

        if st.button("Сгенерировать описание"):
            token = get_token(auth)
            
            # Генерация запроса для GigaChat без индекса
            if description_style == "Краткое":
                description = f"Сделай краткое описание товара: {object_name}. Примерно 4 слова."
            elif description_style == "Техническое":
                description = f"Сделай техническое описание товара: {object_name}. Примерно 10 слов."
            elif description_style == "Подробное":
                description = f"Сделай подробное описание товара: {object_name}. Примерно 20 слов."
            elif description_style == "Рекламное":
                description = f"Создай рекламное описание товара: {object_name}. Примерно 15 слов."
            elif description_style == "Креативное":
                description = f"Напиши креативное описание товара: {object_name}. Примерно 15 слов."
            else:  # Маркетинговое
                description = f"Сделай маркетинговое описание товара: {object_name}. Примерно 15 слов."

            response = get_chat_completion(token, description)
            generated_description = response['choices'][0]['message']['content']

            # Убираем индекс объекта из описания 
            generated_description = generated_description.replace(f"{predicted_idx.item()}", "").strip()  
            st.session_state['generated_description'] = generated_description
            st.write(f"Сгенерированное описание: {generated_description}")
            
        if st.button("Показать итоговый результат"):
            if 'result_image' in st.session_state and 'generated_description' in st.session_state:
                output_path = "final_output.png"
                
                # Исправляем строку с ошибкой
                if background_image is not None:
                    background_image = Image.fromarray(background_image).convert("RGB")  # Преобразуем numpy массив в изображение PIL и конвертируем в RGB
                else:
                    background_color = hex_to_rgb(background_color)

                add_text_above_image(
                    Image.fromarray(st.session_state['result_image']),
                    st.session_state['generated_description'],
                    output_path,
                    font_path=font_file,
                    background_color=background_color,
                    text_color=text_color,
                    font_size=font_size
                )
                st.image(output_path, caption="Итоговое изображение с текстом", use_column_width=True)
            else:
                st.warning("Сначала загрузите изображение и сгенерируйте описание.")

if __name__ == "__main__":
    main()
