import kagglehub
import zipfile
import os

# تنزيل البيانات من Kaggle
path = kagglehub.dataset_download("fidachaudhary/rice-classification")
print("Downloaded file:", path)

# استخراج الملفات
extract_to = 'rice_dataset'
os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("تم استخراج البيانات في:", extract_to)
