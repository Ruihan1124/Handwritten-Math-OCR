import zipfile
import os

# change ZIP file path
zip_path = r'C:\Users\LiRu771\Downloads\archive (2).zip' # zip file path
extract_path = r'C:\Users\LiRu771\PycharmProjects\Handwritten Math OCR\data'  # save file path

# make sure the file exist
os.makedirs(extract_path, exist_ok=True)

#  ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted successfully!")