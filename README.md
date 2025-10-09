# Proyecto de Microexpresiones

Este proyecto permite el preprocesamiento, anotación, extracción de características y entrenamiento de un modelo SVM para la detección de microexpresiones faciales, utilizando datasets como **CASME2** y **SAMM_v1**.

---

## 📂 Estructura del Proyecto

```
Microexpresiones/
│
├── annotations/          # Archivos de anotaciones procesadas
├── data_raw/             # Datasets originales (NO se suben a GitHub)
│   ├── CASME2/
│   └── SAMM_v1/
├── features/             # Features generados a partir de los videos
├── models/               # Modelos entrenados (ej. SVM)
│   └── svm_model.joblib
├── rois/                 # Regiones de interés (ROIs) procesadas
├── src/                  # Scripts principales
│   ├── check_env.py
│   ├── compute_features.py
│   ├── infer_camera.py
│   ├── inspect_casme2.py
│   ├── inspect_samm.py
│   ├── make_casme_annotations.py
│   ├── make_samm_annotations.py
│   ├── preprocess_videos.py
│   ├── train_svm.py
│   └── utils.py
├── config.yaml           # Configuración del proyecto
├── requirements.txt      # Dependencias del entorno
└── .gitignore            # Archivos y carpetas ignoradas por Git
```

---

## ⚙️ Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/SofiaAlvarezS/Microexpresiones.git
cd Microexpresiones
```

2. Crea y activa un entorno virtual:

```bash
python -m venv .venv
# En Windows (PowerShell)
.venv\Scripts\activate
# En Linux/Mac
source .venv/bin/activate
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

4. (Opcional) Verifica el entorno:

```bash
python src/check_env.py
```

---

## 📂 Datasets

Este proyecto trabaja con datasets de microexpresiones como **CASME2** y **SAMM_v1**.  
Debes organizarlos en la carpeta `data_raw/` de la siguiente manera:

```
data_raw/
│
├── CASME2/
│   ├── ... videos e imágenes originales ...
│
└── SAMM_v1/
    ├── ... videos e imágenes originales ...
```

### ⚠️ Importante
- Los datasets **NO están incluidos en este repositorio** por su tamaño y restricciones de licencia.  
- **No subas los datasets a GitHub**. La carpeta `data_raw/` está listada en `.gitignore`, así que Git no los rastrea.  
- Cada miembro del grupo debe descargar los datasets manualmente desde las fuentes oficiales o desde un link compartido (Google Drive, OneDrive, Mega, etc.) y colocarlos dentro de `data_raw/`.

### 📝 Fuentes oficiales
- **CASME2**: [http://casme2.com](http://casme2.com)  
- **SAMM**: [https://sammlab.com](https://sammlab.com)  

---

## ▶️ Uso

### 1. Preprocesar videos
```bash
python src/preprocess_videos.py
```

### 2. Generar anotaciones
```bash
python src/make_casme_annotations.py
python src/make_samm_annotations.py
```

### 3. Extraer características
```bash
python src/compute_features.py
```

### 4. Entrenar el modelo
```bash
python src/train_svm.py
```

### 5. Inferencia en cámara en vivo
```bash
python src/infer_camera.py
```

---

## 📌 Notas adicionales

- La carpeta `rois/` se genera automáticamente con las regiones de interés extraídas de los videos.  
- Si ya tienes el modelo `svm_model.joblib` entrenado en `models/`, no necesitas volver a entrenar a menos que quieras actualizarlo con nuevos datos.  
- Asegúrate de que tu entorno esté activado cada vez que ejecutes un script (`.venv`).
-   No subas datasets completos al repo.
    Se recomienda mantenerlos locales o usar Git LFS para archivos
    pesados.
-   Ajusta rutas y parámetros en config.yaml.

------------------------------------------------------------------------
