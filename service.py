import numpy as np
import io
import json
import logging
import uvicorn
import time
from fastapi import FastAPI, File, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image
from datacontract.service_config import ServiceConfig
from datacontract.service_output import *
from ultralytics import YOLO
import torch
from torchvision import transforms
import pydantic
from ultralytics.utils.plotting import Annotator, colors

# Настройка логгера
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Инициализация FastAPI
app = FastAPI()

# Загрузка конфигурации
service_config_path = r"configs/service_config.json"
with open(service_config_path, "r") as service_config:
    service_config_json = json.load(service_config)
service_config_adapter = pydantic.TypeAdapter(ServiceConfig)
service_config_python = service_config_adapter.validate_python(service_config_json)

# Классы дорожных знаков
class_names = {
    0: "Additional information signs",
    1: "Car",
    2: "Forbidding signs",
    3: "Information signs",
    4: "Prelimainary sings",
    5: "Priority sings",
    6: "Warning sings"
}

# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка классификатора
classifier = torch.load(r'resnet101_best_loss.pth', map_location=device)
classifier.to(device)
classifier.eval()

# Преобразование изображений для классификатора
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

logger.info(f"Загружен классификатор: {service_config_python.path_to_classifier}")

# Загрузка моделей YOLO
logger.info("Загрузка моделей YOLO")
detector_signs = YOLO(r"best.pt")
detector_cars = YOLO(r"yolov8s.pt")
logger.info("Модели загружены")

# Функция классификации
def classify_batch(images: list[Image.Image]) -> list[str]:
    tensor_batch = torch.stack([transform(img) for img in images]).to(device)
    with torch.no_grad():
        outputs = classifier(tensor_batch)
    _, predicted_indices = torch.max(outputs, 1)
    return [class_names[idx.item()] for idx in predicted_indices]

# Проверка состояния сервера
@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Проверка состояния сервиса",
    response_description="Возвращает HTTP статус 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def health_check() -> str:
    return '{"Status" : "OK"}'

# Основной маршрут обработки изображения
@app.post("/file")
async def inference(image: UploadFile = File(...)) -> JSONResponse:
    start_time_ns = time.perf_counter_ns()

    # Чтение изображения
    image_content = await image.read()
    pil_image = Image.open(io.BytesIO(image_content))
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    cv_image = np.array(pil_image)
    logger.info(f"Принята картинка размерности: {cv_image.shape}")

    output_dict = {"objects": []}
    annotator = Annotator(cv_image.copy(), line_width=2)

    # Детекция знаков
    results_signs = detector_signs.predict(cv_image, conf=0.5, verbose=False)
    boxes_signs = results_signs[0].boxes.xyxy
    clss_signs = results_signs[0].boxes.cls

    if boxes_signs is not None and boxes_signs.shape[0] > 0:
        boxes_signs = boxes_signs.cpu().numpy()
        clss_signs = clss_signs.cpu().numpy()

        crop_images = [
            Image.fromarray(cv_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
            for box in boxes_signs
        ]

        class_names_list = classify_batch(crop_images)

        for i, (box, cls, class_name) in enumerate(zip(boxes_signs, clss_signs, class_names_list)):
            annotator.box_label(box, color=colors(int(cls), True), label=class_name)
            output_dict["objects"].append(
                DetectedObject(
                    xtl=int(box[0]), ytl=int(box[1]),
                    xbr=int(box[2]), ybr=int(box[3]),
                    class_name=class_name,
                    tracked_id=i
                )
            )

    # Детекция машин
    results_cars = detector_cars.predict(cv_image, conf=0.5, verbose=False)
    boxes_cars = results_cars[0].boxes.xyxy
    clss_cars = results_cars[0].boxes.cls
    names_coco = results_cars[0].names

    if boxes_cars is not None and boxes_cars.shape[0] > 0:
        boxes_cars = boxes_cars.cpu().numpy()
        clss_cars = clss_cars.cpu().numpy()

        for i, (box, cls) in enumerate(zip(boxes_cars, clss_cars)):
            class_name = names_coco[int(cls)]
            if class_name.lower() == "car":
                annotator.box_label(box, color=(0, 255, 0), label=class_name)
                output_dict["objects"].append(
                    DetectedObject(
                        xtl=int(box[0]), ytl=int(box[1]),
                        xbr=int(box[2]), ybr=int(box[3]),
                        class_name="Car",
                        tracked_id=1000 + i
                    )
                )


    # Формирование JSON
    service_output = ServiceOutput(objects=output_dict["objects"])
    service_output_json = service_output.model_dump(mode="json")

    # Сохранение JSON в файл
    with open("output_json.json", "w") as output_file:
        json.dump(service_output_json, output_file, indent=4)

    elapsed_us = (time.perf_counter_ns() - start_time_ns) / 1000  # время в микросекундах
    logger.info(f"Обнаружено объектов: {len(output_dict['objects'])}")
    logger.info(f"Время обработки: {elapsed_us:.2f} мкс")

    response = JSONResponse(content=jsonable_encoder(service_output_json))
    response.headers["X-Process-Time-us"] = f"{elapsed_us:.2f}"
    return response

# Запуск сервера
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
