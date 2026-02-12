import cv2
import os
import sys
import json
from app.services.ocr_service import OcrService

def test_ocr(image_path):
    print(f"Testing OCR on: {image_path}")
    image = cv2.imread(image_path)
    results = OcrService.process(image)
    fields = [
        "nik", "nama", "tempat_lahir", "tgl_lahir", "jenis_kelamin",
        "gol_darah", "agama", "status_perkawinan", "provinsi", "kabupaten",
        "alamat", "rt_rw", "kel_desa", "kecamatan", "pekerjaan", "kewarganegaraan"
    ]
    output = dict(zip(fields, results))
    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    test_ocr("/Users/eric/Documents/Project/python/rnd/ktp_ocr/test/sample/1.jpg")
