# KTP OCR API (FastAPI)

A high-performance API for extracting data (NIK, Nama, TTL, etc.) from Indonesian ID Cards (KTP) using OCR and Deep Learning.

Built with **FastAPI**, **OpenCV**, and **Tesseract OCR**.

## Features

- **OCR Extraction**: Automatically extracts NIK, Name, Date of Birth, Blood Type, Gender, Religion, Marital Status, Occupation, Nationality, and Address.
- **Advanced Processing**: Uses CLAHE, Thresholding, and strict validation logic to ensure high accuracy (including robust Blood Type extraction).
- **KTP Detection**: (Optional) CNN-based classification to verify if the uploaded image is a KTP.
- **FastAPI Powered**: Async, high-performance, and auto-generated Swagger documentation.

## Prerequisites

- **Python 3.10+**
- **Tesseract OCR** installed on your system:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr tesseract-ocr-ind libtesseract-dev

  # macOS
  brew install tesseract
  brew install tesseract-lang
  ```

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/ktp-ocr.git
    cd ktp-ocr
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup Environment Variables**
    Create a `.env` file (optional, defaults are set in code):
    ```env
    PORT=8000
    DEBUG=True
    ```

## Running the Application

Start the server using Uvicorn:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

## API Documentation

Interactive documentation is available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### OCR Endpoint

**POST** `/api/ocr/`

- **Request**: `multipart/form-data`
    - `image`: The KTP image file (JPG, PNG).

- **Response Example**:
    ```json
    {
      "meta": {
        "code": 200,
        "message": "Proses OCR Berhasil",
        "error": false
      },
      "data": {
        "nik": "1234567890123456",
        "nama": "JOHN DOE",
        "tempat_lahir": "JAKARTA",
        "tgl_lahir": "01-01-1990",
        "jenis_kelamin": "LAKI-LAKI",
        "gol_darah": "O",
        "agama": "ISLAM",
        "status_perkawinan": "BELUM KAWIN",
        "pekerjaan": "PEGAWAI SWASTA",
        "kewarganegaraan": "WNI",
        "alamat": {
          "name": "JL. CONTOH ALAMAT NO. 123",
          "rt_rw": "001/002",
          "kel_desa": "KELURAHAN CONTOH",
          "kecamatan": "KECAMATAN CONTOH",
          "kabupaten": "JAKARTA SELATAN",
          "provinsi": "DKI JAKARTA"
        },
        "time_elapsed": "1.234"
      }
    }
    ```

## Project Structure

```
ktp_ocr/
├── app/
│   ├── routes/         # API Endpoints
│   ├── services/       # Business Logic (OCR, CNN)
│   ├── models/         # Pydantic Schemas
│   └── handler/        # Error Handling
├── data/               # Models and Data
├── images/             # Sample images
├── main.py             # Entrypoint
└── requirements.txt
```

## Acknowledgments

- Based on original work by [enningxie/KTP-OCR](https://github.com/enningxie/KTP-OCR) and [jeffreyevan/OCR](https://github.com/jeffreyevan/OCR).
