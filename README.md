Microexpresiones

Este repositorio contiene un proyecto de detección y análisis de
microexpresiones faciales.
Incluye scripts para preprocesamiento de videos, anotaciones, extracción
de características, entrenamiento de un modelo SVM y pruebas de
inferencia en cámara.

------------------------------------------------------------------------

📂 Estructura del proyecto

    ├── .venv/                   # Entorno virtual (ignorado en git)
    ├── annotations/             # Archivos de anotaciones (labels)
    ├── data_raw/                # Dataset original (videos/imágenes sin procesar)
    ├── features/                # Características extraídas (features)
    ├── models/                  # Modelos entrenados (ej: svm_model.joblib)
    ├── rois/                    # Regiones de interés (imagenes recortadas)
    ├── src/                     # Código fuente
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
    ├── .gitignore
    ├── config.yaml              # Configuración general
    ├── requirements.txt         # Dependencias necesarias

------------------------------------------------------------------------

⚙️ Instalación

1.  Clonar el repositorio:

        git clone https://github.com/SofiaAlvarezS/Microexpresiones.git
        cd Microexpresiones

2.  Crear entorno virtual (recomendado):

        python -m venv .venv

3.  Activar el entorno virtual:

    -   En Windows (PowerShell):

            .venv\Scripts\activate

    -   En Linux/Mac:

            source .venv/bin/activate

4.  Instalar dependencias:

        pip install -r requirements.txt

5.  Verificar instalación:

        python src/check_env.py

------------------------------------------------------------------------

📊 Uso de datasets

-   Coloca tus videos o imágenes originales en data_raw/.
-   El preprocesamiento extraerá ROIs en la carpeta rois/.
    -   Si no existe, créala:

            mkdir rois
-   Las features procesadas se guardarán en features/.
-   Los modelos entrenados se guardan en models/.

⚠️ Nota: Ya hay un modelo entrenado (svm_model.joblib). Puedes usarlo
directamente o reentrenar con nuevos datos.

------------------------------------------------------------------------

🚀 Cómo usar los scripts

1.  Preprocesar videos:

        python src/preprocess_videos.py

2.  Generar anotaciones (ejemplos con datasets específicos):

        python src/make_casme_annotations.py
        python src/make_samm_annotations.py

3.  Extraer características:

        python src/compute_features.py

4.  Entrenar modelo SVM:

        python src/train_svm.py

5.  Inferencia con cámara en vivo:

        python src/infer_camera.py

------------------------------------------------------------------------

📌 Notas

-   No subas datasets completos al repo.
    Se recomienda mantenerlos locales o usar Git LFS para archivos
    pesados.

-   Ajusta rutas y parámetros en config.yaml.

------------------------------------------------------------------------
