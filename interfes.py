import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import requests
import time
from datetime import timedelta
import os
import json
from collections import defaultdict

API_URL = "http://localhost:8000/file"

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Детекция объектов на видео")
        self.root.geometry("1080x800")
        self.root.configure(bg="#1e1e1e")

        # --- Стили ---
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self.style.configure("TButton", font=("Segoe UI", 11), padding=6, background="#2e2e2e", foreground="white")
        self.style.configure("TLabel", font=("Segoe UI", 10), background="#1e1e1e", foreground="white")
        self.style.configure("TCheckbutton", font=("Segoe UI", 10), background="#1e1e1e", foreground="white")
        self.style.configure("TScale", background="#1e1e1e")

        # --- Верхняя панель с загрузкой ---
        load_frame = ttk.Frame(root)
        load_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.btn_load = ttk.Button(load_frame, text="📂 Загрузить видео", command=self.load_video)
        self.btn_load.pack(side=tk.LEFT)

        # --- Canvas для видео ---
        self.canvas = tk.Canvas(root, bg="black", width=850, height=500)
        self.canvas.pack(padx=10, pady=10, expand=True)

        # --- Слайдер и тайм-код ---
        slider_frame = ttk.Frame(root)
        slider_frame.pack(fill=tk.X, padx=10)

        self.slider = ttk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.slider.pack(fill=tk.X)

        self.label_frame_info = ttk.Label(root, text="Кадр: 0 / 0 | Время: 00:00 / 00:00", anchor="center")
        self.label_frame_info.pack(pady=(0, 5))

        # --- Панель управления воспроизведением ---
        control_frame = ttk.Frame(root)
        control_frame.pack(pady=5)

        self.btn_prev = ttk.Button(control_frame, text="⏪ Назад", command=self.prev_frame, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_start = ttk.Button(control_frame, text="▶ Старт", command=self.start_playback, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_pause = ttk.Button(control_frame, text="⏸ Пауза", command=self.pause_playback, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=5)

        self.btn_next = ttk.Button(control_frame, text="⏩ Вперед", command=self.next_frame, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        self.btn_save = ttk.Button(control_frame, text="💾 Сохранить видео", command=self.save_video, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.btn_save_segment = ttk.Button(control_frame, text="✂ Сохранить отрезок", command=self.save_video_segment, state=tk.DISABLED)
        self.btn_save_segment.pack(side=tk.LEFT, padx=5)

        self.btn_generate_report = ttk.Button(control_frame, text="📄 Отчет", command=self.generate_report, state=tk.DISABLED)
        self.btn_generate_report.pack(side=tk.LEFT, padx=5)

        # --- Фильтры классов ---
        filter_frame = ttk.LabelFrame(root, text="Фильтры объектов")
        filter_frame.pack(fill=tk.X, padx=10, pady=5)

        self.available_classes = ["Additional information signs", "Car", "Forbidding signs", "Information signs", "Preliminary signs", "Priority signs", "Warning sings"]
        self.class_vars = {}
        self.class_filters = set(self.available_classes)
        self.detection_data = []

        for cls in self.available_classes:
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(filter_frame, text=cls, variable=var, command=self.update_class_filters)
            chk.pack(side=tk.LEFT, padx=5, pady=2)
            self.class_vars[cls] = var

        # --- Статус бар ---
        self.status_bar = ttk.Label(root, text="Готов к работе", anchor="w", relief=tk.FLAT,
                                    background="#111", foreground="white")
        self.status_bar.pack(fill=tk.X, padx=10, pady=(5, 0))

        # --- Видео переменные ---
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.duration = 0
        self.frame_width = 0
        self.frame_height = 0
        self.current_frame_idx = 0
        self.frame_cache = []
        self.playing = False
        self.thread = None
        self.last_frame_time = 0
        self.output_video = None

    
    def update_status(self, message):
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Видео файлы", "*.mp4 *.avi *.mov *.mkv"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        
        self.stop_playback()
        
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть видео")
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0
        self.frame_cache = [None] * self.total_frames
        self.detection_data = []  # Очищаем данные детекции
        
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_prev.config(state=tk.NORMAL if self.total_frames > 1 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.total_frames > 1 else tk.DISABLED)
        self.btn_save.config(state=tk.NORMAL)
        self.btn_save_segment.config(state=tk.NORMAL)
        self.btn_generate_report.config(state=tk.NORMAL)
        self.update_frame_info()
        self.show_frame_by_index(self.current_frame_idx)
        self.slider.config(to=self.total_frames - 1)
        self.slider.configure(state='normal')
        
        message = (f"Загружено: {os.path.basename(self.video_path)} | "
                  f"Кадров: {self.total_frames} | "
                  f"FPS: {self.fps:.2f} | "
                  f"Размер: {self.frame_width}x{self.frame_height} | "
                  f"Длительность: {str(timedelta(seconds=self.duration))}")
        self.update_status(message)

    def update_class_filters(self):
        self.class_filters = {cls for cls, var in self.class_vars.items() if var.get()}
        self.show_frame_by_index(self.current_frame_idx)
   
    def save_video_segment(self):
        if not self.video_path or not self.cap:
            messagebox.showwarning("Ошибка", "Сначала загрузите видео")
            return
        
        # Создаем окно для ввода временного отрезка
        segment_window = tk.Toplevel(self.root)
        segment_window.title("Сохранение отрезка видео")
        segment_window.geometry("400x200")
        
        ttk.Label(segment_window, text="Начало отрезка (мм:сс):").pack(pady=5)
        start_entry = ttk.Entry(segment_window)
        start_entry.pack(pady=5)
        
        ttk.Label(segment_window, text="Конец отрезка (мм:сс):").pack(pady=5)
        end_entry = ttk.Entry(segment_window)
        end_entry.pack(pady=5)
        
        def save_segment():
            try:
                # Парсим введенное время
                start_min, start_sec = map(int, start_entry.get().split(':'))
                end_min, end_sec = map(int, end_entry.get().split(':'))
                
                start_time = start_min * 60 + start_sec
                end_time = end_min * 60 + end_sec
                
                if start_time >= end_time:
                    messagebox.showerror("Ошибка", "Время начала должно быть меньше времени конца")
                    return
                
                if end_time > self.duration:
                    messagebox.showerror("Ошибка", "Конец отрезка превышает длительность видео")
                    return
                
                # Запрашиваем путь для сохранения
                output_path = filedialog.asksaveasfilename(
                    defaultextension=".mp4",
                    filetypes=[("MP4 файлы", "*.mp4"), ("AVI файлы", "*.avi"), ("Все файлы", "*.*")],
                    initialfile=f"segment_{start_min}_{start_sec}-{end_min}_{end_sec}.mp4"
                )
                
                if not output_path:
                    return
                
                # Определяем кодек
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') if output_path.endswith('.mp4') else cv2.VideoWriter_fourcc(*'XVID')
                
                # Рассчитываем кадры для отрезка
                start_frame = int(start_time * self.fps)
                end_frame = int(end_time * self.fps)
                
                self.update_status(f"Сохранение отрезка {start_time}s-{end_time}s...")
                self.root.update()
                
                # Создаем объект для записи видео
                out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
                
                # Обрабатываем и сохраняем каждый кадр отрезка
                for i in range(start_frame, end_frame + 1):
                    frame = self.get_frame(i)
                    if frame is None:
                        continue
                    
                    annotated_frame = self.annotate_frame(frame.copy())
                    out.write(annotated_frame)
                    
                    # Обновляем прогресс
                    if i % 10 == 0:
                        progress = (i - start_frame + 1) / (end_frame - start_frame + 1) * 100
                        self.update_status(f"Сохранение отрезка... {progress:.1f}% завершено")
                        self.root.update()
                
                out.release()
                self.update_status(f"Отрезок успешно сохранен: {os.path.basename(output_path)}")
                messagebox.showinfo("Успех", f"Отрезок видео успешно сохранен:\n{output_path}")
                segment_window.destroy()
                
            except Exception as e:
                self.update_status("Ошибка при сохранении отрезка")
                messagebox.showerror("Ошибка", f"Не удалось сохранить отрезок:\n{str(e)}")
                if 'out' in locals():
                    out.release()
        
        ttk.Button(segment_window, text="Сохранить", command=save_segment).pack(pady=10)
    
    def generate_report(self):
        if not self.video_path or not self.cap:
            messagebox.showwarning("Ошибка", "Сначала загрузите видео")
            return
        
        # Анализируем данные детекции по минутам
        report_data = defaultdict(lambda: defaultdict(int))
        
        for frame_data in self.detection_data:
            frame_time = frame_data['frame_time']
            minute = int(frame_time // 60)
            
            for obj in frame_data['objects']:
                class_name = obj['class_name']
                report_data[minute][class_name] += 1
        
        # Создаем текстовый отчет
        report_text = "Отчет по видео:\n\n"
        report_text += f"Видео: {os.path.basename(self.video_path)}\n"
        report_text += f"Длительность: {str(timedelta(seconds=self.duration))}\n\n"
        report_text += "Детекция по минутам:\n"
        
        for minute in sorted(report_data.keys()):
            report_text += f"\nМинута {minute}:\n"
            for class_name, count in report_data[minute].items():
                report_text += f"  {class_name}: {count}\n"
        
        # Создаем JSON структуру
        report_json = {
            "video_file": os.path.basename(self.video_path),
            "duration": str(timedelta(seconds=self.duration)),
            "detection_data": {f"minute_{minute}": dict(report_data[minute]) 
                            for minute in sorted(report_data.keys())}
        }
        
        # Создаем CSV данные
        csv_lines = ["Минута,Класс,Количество"]
        for minute in sorted(report_data.keys()):
            for class_name, count in report_data[minute].items():
                csv_lines.append(f"{minute},{class_name},{count}")
        report_csv = "\n".join(csv_lines)
        
        # Показываем отчет в новом окне
        report_window = tk.Toplevel(self.root)
        report_window.title("Отчет по видео")
        report_window.geometry("700x550")
        
        # Фрейм для выбора формата
        format_frame = ttk.Frame(report_window)
        format_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(format_frame, text="Формат отчета:").pack(side=tk.LEFT)
        self.report_format = tk.StringVar(value="txt")
        ttk.Radiobutton(format_frame, text="TXT", variable=self.report_format, value="txt").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="JSON", variable=self.report_format, value="json").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="CSV", variable=self.report_format, value="csv").pack(side=tk.LEFT, padx=5)
        
        text_frame = ttk.Frame(report_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.text_widget = tk.Text(text_frame, wrap=tk.WORD)
        self.text_widget.insert(tk.END, report_text)
        self.text_widget.config(state=tk.DISABLED)
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Функция обновления отчета при смене формата
        def update_report_display(*args):
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.delete(1.0, tk.END)
            
            if self.report_format.get() == "txt":
                self.text_widget.insert(tk.END, report_text)
            elif self.report_format.get() == "json":
                self.text_widget.insert(tk.END, json.dumps(report_json, indent=4, ensure_ascii=False))
            elif self.report_format.get() == "csv":
                self.text_widget.insert(tk.END, report_csv)
            
            self.text_widget.config(state=tk.DISABLED)
        
        self.report_format.trace_add("write", update_report_display)
        
        # Функция сохранения отчета
        def save_report():
            file_types = {
                "txt": [("Текстовые файлы", "*.txt")],
                "json": [("JSON файлы", "*.json")],
                "csv": [("CSV файлы", "*.csv")]
            }
            
            default_ext = {
                "txt": ".txt",
                "json": ".json",
                "csv": ".csv"
            }
            
            fmt = self.report_format.get()
            file_path = filedialog.asksaveasfilename(
                defaultextension=default_ext[fmt],
                filetypes=file_types[fmt],
                initialfile=f"report_{os.path.splitext(os.path.basename(self.video_path))[0]}.{fmt}"
            )
            
            if not file_path:
                return
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    if fmt == "txt":
                        f.write(report_text)
                    elif fmt == "json":
                        json.dump(report_json, f, indent=4, ensure_ascii=False)
                    elif fmt == "csv":
                        f.write(report_csv)
                
                messagebox.showinfo("Успех", f"Отчет сохранен в формате {fmt.upper()}:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить отчет:\n{str(e)}")
        
        # Кнопка сохранения
        ttk.Button(report_window, text="Сохранить отчет", command=save_report).pack(pady=10)
    def save_video(self):
        if not self.video_path or not self.cap:
            messagebox.showwarning("Ошибка", "Сначала загрузите видео")
            return
        
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 файлы", "*.mp4"), ("AVI файлы", "*.avi"), ("Все файлы", "*.*")],
            initialfile="output_" + os.path.basename(self.video_path)
        )
        
        if not output_path:
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if output_path.endswith('.mp4') else cv2.VideoWriter_fourcc(*'XVID')
        
        try:
            self.update_status("Сохранение видео... Пожалуйста, подождите")
            self.root.update()
            
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
            
            for i in range(self.total_frames):
                frame = self.get_frame(i)
                if frame is None:
                    continue
                
                annotated_frame = self.annotate_frame(frame.copy())
                out.write(annotated_frame)
                
                if i % 10 == 0:
                    progress = (i + 1) / self.total_frames * 100
                    self.update_status(f"Сохранение видео... {progress:.1f}% завершено")
                    self.root.update()
            
            out.release()
            self.update_status(f"Видео успешно сохранено: {os.path.basename(output_path)}")
            messagebox.showinfo("Успех", f"Видео успешно сохранено:\n{output_path}")
            
        except Exception as e:
            self.update_status("Ошибка при сохранении видео")
            messagebox.showerror("Ошибка", f"Не удалось сохранить видео:\n{str(e)}")
            if 'out' in locals():
                out.release()
    
    def start_playback(self):
        if self.playing:
            return
        if self.cap is None:
            messagebox.showwarning("Видео не загружено", "Сначала загрузите видео")
            return
        
        self.playing = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_load.config(state=tk.DISABLED)
        self.btn_prev.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.btn_save_segment.config(state=tk.DISABLED)
        self.btn_generate_report.config(state=tk.DISABLED)
        
        self.thread = threading.Thread(target=self.play_video_thread, daemon=True)
        self.thread.start()
    
    def play_video_thread(self):
        while self.playing and self.current_frame_idx < self.total_frames:
            start_time = time.time()
            
            frame = self.get_frame(self.current_frame_idx)
            if frame is None:
                break

            annotated = self.annotate_frame(frame.copy())
            
            self.root.after(0, lambda: self.show_image(annotated))
            self.root.after(0, self.update_frame_info)
            self.root.after(0, lambda: self.slider.set(self.current_frame_idx))

            self.current_frame_idx += 1
            
            frame_time = 1 / self.fps if self.fps > 0 else 1/30
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

        self.root.after(0, self.pause_playback)
    
    def pause_playback(self):
        if not self.playing:
            return
        self.playing = False
        if self.thread is not None:
            self.thread.join(timeout=1)
        self.thread = None
        
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_load.config(state=tk.NORMAL)
        self.btn_prev.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)
        self.btn_save_segment.config(state=tk.NORMAL)
        self.btn_generate_report.config(state=tk.NORMAL)
    
    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx = max(0, self.current_frame_idx - 1)
            self.show_frame_by_index(self.current_frame_idx)
    
    def next_frame(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
            self.show_frame_by_index(self.current_frame_idx)
    
    def on_slider_move(self, val):
        frame_idx = int(float(val))
        if frame_idx != self.current_frame_idx:
            self.current_frame_idx = frame_idx
            self.show_frame_by_index(frame_idx)
            self.update_frame_info()

    def update_frame_info(self):
        current_time = self.current_frame_idx / self.fps if self.fps > 0 else 0
        total_time = self.duration
        
        current_time_str = str(timedelta(seconds=current_time)).split(".")[0]
        total_time_str = str(timedelta(seconds=total_time)).split(".")[0]
        
        self.label_frame_info.config(
            text=f"Кадр: {self.current_frame_idx+1} / {self.total_frames} | "
                 f"Время: {current_time_str} / {total_time_str}"
        )
    
    def get_frame(self, index):
        if index < 0 or index >= self.total_frames:
            return None
        if self.frame_cache[index] is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = self.cap.read()
            if ret:
                self.frame_cache[index] = frame
        return self.frame_cache[index]
    
    def show_frame_by_index(self, index):
        frame = self.get_frame(index)
        if frame is None:
            return
        
        self.current_frame_idx = index
        self.update_frame_info()
        
        if int(self.slider.get()) != index:
            self.slider.set(index)

        annotated = self.annotate_frame(frame.copy())
        self.show_image(annotated)
            
    def annotate_frame(self, frame):
        try:
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
            response = requests.post(API_URL, files=files)
            response.raise_for_status()
            data = response.json()

            # Сохраняем данные детекции
            frame_time = self.current_frame_idx / self.fps if self.fps > 0 else 0
            frame_data = {
                'frame_idx': self.current_frame_idx,
                'frame_time': frame_time,
                'objects': data.get("objects", [])
            }
            self.detection_data.append(frame_data)

            # Размер кадра
            h, w, _ = frame.shape

            for obj in data.get("objects", []):
                label = obj['class_name']

                # Временно отключаем фильтрацию классов
                if label not in self.class_filters:
                    continue

                # Координаты — обрезаем по границам кадра
                xtl = max(0, min(int(obj['xtl']), w - 1))
                ytl = max(0, min(int(obj['ytl']), h - 1))
                xbr = max(0, min(int(obj['xbr']), w - 1))
                ybr = max(0, min(int(obj['ybr']), h - 1))

                # Отрисовка прямоугольника и подписи
                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
                cv2.putText(frame, label, (xtl, max(ytl - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        except Exception as e:
            print("Ошибка при запросе к серверу:", e)

        return frame

    
    def show_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img.thumbnail((canvas_w, canvas_h))
        
        imgtk = ImageTk.PhotoImage(img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(canvas_w//2, canvas_h//2, anchor=tk.CENTER, image=imgtk)
    
    def stop_playback(self):
        self.pause_playback()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.frame_cache.clear()
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 0
        self.duration = 0
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_prev.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.btn_save_segment.config(state=tk.DISABLED)
        self.btn_generate_report.config(state=tk.DISABLED)
        self.update_frame_info()
        self.slider.config(to=0)
        self.slider.set(0)
        self.slider.configure(state='disabled')
        self.canvas.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()