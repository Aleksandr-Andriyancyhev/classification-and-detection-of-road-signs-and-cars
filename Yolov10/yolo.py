from ultralytics import YOLO

def main():
    model = YOLO('yolov10n.pt')  # можно поменять на yolov10n.pt, yolov10m.pt и т.д.

    # Обучение модели

    model.train(
        data='set1.yaml',
        epochs=100,
        imgsz=720,
        batch=8,
        device=0,
        project="yolo_train",
        name="test1",
        patience=15,

        # Аугментации
        hsv_h=0.01,                 # меньше искажения цвета
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=0.0,                # по-прежнему не вращаем
        translate=0.05,             # меньше сдвигов
        scale=0.3,                  # умеренное масштабирование
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.7,                 # немного уменьшаем силу мозаики
        mixup=0.0,                  # отключим — часто вредит при малых датасетах
        copy_paste=0.0              # тоже отключим
    )


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # важно для Windows
    main()