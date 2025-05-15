import nltk
import os
import ssl

# Bypass SSL verification (temporary workaround for download)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download to a temporary directory
temp_dir = "D:/VSCode Projects/RAG Project/nltk_temp"
os.makedirs(temp_dir, exist_ok=True)
nltk.data.path.append(temp_dir)
nltk.download('punkt', download_dir=temp_dir)
nltk.download('punkt_tab', download_dir=temp_dir)
print(f"Data downloaded to {temp_dir}")