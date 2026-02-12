import cv2
import os
import sys
import json
from app.services.ocr_service import OcrService

def test_ocr(image_path):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return

    print(f"Testing OCR on: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not decode image")
        return

    try:
        results = OcrService.process(image)
        fields = [
            "nik", "nama", "tempat_lahir", "tgl_lahir", "jenis_kelamin",
            "gol_darah", "agama", "status_perkawinan", "provinsi", "kabupaten",
            "alamat", "rt_rw", "kel_desa", "kecamatan", "pekerjaan", "kewarganegaraan"
        ]
        
        output = dict(zip(fields, results))
        print(json.dumps(output, indent=4))
        
    except Exception as e:
        print(f"OCR failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    sample_dir = "/Users/eric/Documents/Project/python/rnd/ktp_ocr/test/sample"
    images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    for img_name in sorted(images):
        img_path = os.path.join(sample_dir, img_name)
        test_ocr(img_path)
        print("-" * 50)
