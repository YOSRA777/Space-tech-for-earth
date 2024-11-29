import kagglehub

# Download latest version
path = kagglehub.dataset_download("noaa/seismic-waves")

print("Path to dataset files:", path)