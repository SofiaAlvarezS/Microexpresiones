import sys
import cv2
import joblib
import numpy as np
import mediapipe as mp
from collections import Counter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QComboBox, QVBoxLayout, QWidget, QFileDialog, QTextEdit, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from src.extract_hof_features import extract_hof_from_sequence
from src.extract_regions import extract_regions


class MicroExpressionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microexpresiones - Interfaz")
        self.setGeometry(100, 100, 1100, 700)

        print("✅ Cargando modelo...")
        model_bundle = joblib.load("models/svm_model.joblib")

        # Cargar modelo + scaler (según cómo se guardó)
        if isinstance(model_bundle, dict):
            self.model = model_bundle["model"]
            self.scaler = model_bundle["scaler"]
        else:
            self.model = getattr(model_bundle, "named_steps", {}).get("model", model_bundle)
            self.scaler = getattr(model_bundle, "named_steps", {}).get("scaler", None)

        # FaceMesh de MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # GUI
        self.label_video = QLabel()
        self.label_video.setFixedSize(800, 500)
        self.label_video.setAlignment(Qt.AlignCenter)

        self.label_roi = QLabel()
        self.label_roi.setFixedSize(250, 250)
        self.label_roi.setAlignment(Qt.AlignCenter)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Cámara", "Video"])

        self.combo_roi = QComboBox()
        self.combo_roi.addItems(["rostro_completo", "ojos_der", "ojos_izq", "cejas", "nariz_boca"])

        self.btn_load_video = QPushButton("🎬 Cargar Video")
        self.btn_start = QPushButton("▶ Iniciar")
        self.btn_pause = QPushButton("⏸ Pausar")
        self.btn_stop = QPushButton("⏹ Detener")

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)

        # Layouts
        layout_video = QHBoxLayout()
        layout_video.addWidget(self.label_video)
        layout_video.addWidget(self.label_roi)

        layout_main = QVBoxLayout()
        layout_main.addLayout(layout_video)
        layout_main.addWidget(self.combo_mode)
        layout_main.addWidget(self.combo_roi)
        layout_main.addWidget(self.btn_load_video)
        layout_main.addWidget(self.btn_start)
        layout_main.addWidget(self.btn_pause)
        layout_main.addWidget(self.btn_stop)
        layout_main.addWidget(self.stats_text)

        container = QWidget()
        container.setLayout(layout_main)
        self.setCentralWidget(container)

        # Variables
        self.cap = None
        self.video_path = None
        self.running = False
        self.predictions = []
        self.frame_counter = 0
        self.stats_update_counter = 0

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Conexiones
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_pause.clicked.connect(self.pause_processing)
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_load_video.clicked.connect(self.load_video)

    # --- Inicialización de cámara/video ---
    def open_camera(self):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                print(f"📷 Cámara inicializada con backend {backend}")
                return cap
        print("❌ No se pudo abrir la cámara.")
        return None

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar video", "", "Videos (*.mp4 *.avi)")
        if path:
            self.video_path = path
            self.combo_mode.setCurrentText("Video")
            print(f"🎬 Video seleccionado: {path}")

    # --- Control del flujo ---
    def start_processing(self):
        mode = self.combo_mode.currentText()
        if mode == "Video" and self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = self.open_camera()

        if not self.cap or not self.cap.isOpened():
            print("❌ No se pudo acceder a la fuente de video.")
            return

        self.running = True
        self.predictions.clear()
        self.frame_counter = 0
        self.stats_text.clear()
        self.timer.start(30)
        print("▶ Procesamiento iniciado.")

    def pause_processing(self):
        self.running = False
        self.timer.stop()
        print("⏸ Procesamiento pausado.")

    def stop_processing(self):
        self.running = False
        self.timer.stop()
        self.show_statistics()
        print("⏹ Procesamiento detenido.")

    # --- Procesamiento de frames ---
    def update_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            print("📊 Fin del video o error de cámara.")
            self.stop_processing()
            return

        frame = cv2.flip(frame, 1)  # 🔄 Corrige cámara invertida
        roi_name = self.combo_roi.currentText()
        self.frame_counter += 1
        pred = "No detectado"

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            regions = extract_regions(rgb, self.face_mesh)

            if roi_name in regions:
                roi = regions[roi_name]
                roi_resized = cv2.resize(roi, (128, 128))
                self.display_roi(roi_resized)

                gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                hof_vec = extract_hof_from_sequence([gray])
                if hof_vec is not None:
                    X = np.array(hof_vec).reshape(1, -1)
                    if self.scaler is not None:
                        X = self.scaler.transform(X)
                    pred = self.model.predict(X)[0]
                    self.predictions.append(pred)

                    # 📊 Actualiza cada 2 seg
                    self.stats_update_counter += 1
                    if self.stats_update_counter % 60 == 0:
                        self.show_statistics(intermediate=True)

            # 🔲 Muestra región seleccionada en el frame
            if roi_name in regions:
                (h, w, _) = frame.shape
                region = regions[roi_name]
                mask = np.zeros_like(frame)
                mask[:region.shape[0], :region.shape[1]] = region
                cv2.rectangle(frame, (30, 30), (w - 30, h - 30), (0, 255, 0), 1)

            cv2.putText(frame, f"Emoción: {pred}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"⚠️ Error en frame: {e}")

        self.display_frame(frame)

    # --- Mostrar imagenes ---
    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(img))

    def display_roi(self, roi):
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.label_roi.setPixmap(QPixmap.fromImage(img))

    # --- Estadísticas ---
    def show_statistics(self, intermediate=False):
        if not self.predictions:
            if not intermediate:
                self.stats_text.setText("⚠️ No se registraron emociones.")
            return

        counts = Counter(self.predictions)
        total = sum(counts.values())
        lines = [f"{emo}: {c/total:.1%}" for emo, c in counts.items()]
        text = "\n".join(lines)

        if not intermediate:
            print("\n📊 Estadísticas finales:")
            for line in lines:
                print("  " + line)
            print("✅ Análisis completado.\n")

        self.stats_text.setText(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MicroExpressionApp()
    win.show()
    sys.exit(app.exec_())
