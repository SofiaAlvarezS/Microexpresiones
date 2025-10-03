import sys
print("Python:", sys.version)

packages = ["cv2", "numpy", "sklearn", "mediapipe", "pandas", "joblib"]
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "no __version__ attr")
        print(f"{pkg}: OK (version {version})")
    except Exception as e:
        print(f"{pkg}: ERROR -> {e}")
