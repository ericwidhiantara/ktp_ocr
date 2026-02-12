import time

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from PIL import Image

from app.models.schemas import BaseResp, Meta, OcrResultSchema, AlamatSchema
from app.services.cnn_service import CnnService
from app.services.ocr_service import OcrService

router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.post(
    "",
    summary="OCR KTP",
    description="Upload a KTP image to extract identity data via OCR",
)
async def ocr_ktp(image: UploadFile = File(..., description="KTP image file")):
    """
    Process a KTP (Indonesian ID card) image and extract identity data.
    """
    start_time = time.time()

    try:
        # Read image bytes
        image_bytes = await image.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        cv_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Convert grayscale to BGR if needed
        if cv_image is not None and len(cv_image.shape) == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        # Check if image is a KTP using CNN
        # Special handling for grayscale: CNN model is color-trained and might reject grayscale KTPs.
        # We check saturation to determine if we should allow a bypass.
        await image.seek(0)
        pil_image = Image.open(image.file)
        pil_image = pil_image.convert("RGB")
        
        # Calculate saturation to detect grayscale
        cv_image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(cv_image_hsv[:, :, 1])
        
        is_ktp = True
        # If saturation is high enough, we trust the CNN model
        if avg_saturation > 5:
            is_ktp = CnnService.is_ktp(pil_image)
            print(f"[OCR Debug] Color image detected (Sat: {avg_saturation:.2f}). CNN classification: {is_ktp}")
        else:
            print(f"[OCR Debug] Grayscale/Low-sat image detected (Sat: {avg_saturation:.2f}). Bypassing CNN check.")

        if not is_ktp:
            return JSONResponse(
                status_code=400,
                content=jsonable_encoder(
                    BaseResp(
                        meta=Meta(
                            code=400,
                            message="Foto yang diunggah haruslah foto E-KTP",
                            error=True,
                        ),
                    )
                ),
            )

        # Process OCR
        (
            nik, nama, tempat_lahir, tgl_lahir, jenis_kelamin, gol_darah,
            agama, status_perkawinan, provinsi, kabupaten, alamat, rt_rw,
            kel_desa, kecamatan, pekerjaan, kewarganegaraan,
        ) = OcrService.process(cv_image)

        finish_time = time.time() - start_time

        # Debug: log parsed fields
        print(f"[OCR Debug] nik={repr(nik)}, nama={repr(nama)}, provinsi={repr(provinsi)}, kabupaten={repr(kabupaten)}")

        # Validate required fields
        if not nik or not nama or not provinsi or not kabupaten:
            return JSONResponse(
                status_code=400,
                content=jsonable_encoder(
                    BaseResp(
                        meta=Meta(
                            code=400,
                            message="Resolusi foto terlalu rendah, silakan coba lagi.",
                            error=True,
                        ),
                    )
                ),
            )

        return BaseResp(
            meta=Meta(
                code=200,
                message="Proses OCR Berhasil",
                error=False,
            ),
            data=OcrResultSchema(
                nik=str(nik),
                nama=str(nama),
                tempat_lahir=str(tempat_lahir),
                tgl_lahir=str(tgl_lahir),
                jenis_kelamin=str(jenis_kelamin),
                gol_darah=str(gol_darah),
                agama=str(agama),
                status_perkawinan=str(status_perkawinan),
                pekerjaan=str(pekerjaan),
                kewarganegaraan=str(kewarganegaraan),
                alamat=AlamatSchema(
                    name=str(alamat),
                    rt_rw=str(rt_rw),
                    kel_desa=str(kel_desa),
                    kecamatan=str(kecamatan),
                    kabupaten=str(kabupaten),
                    provinsi=str(provinsi),
                ),
                time_elapsed=str(round(finish_time, 3)),
            ),
        )

    except Exception as e:
        print(f"Error processing KTP: {e}")
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder(
                BaseResp(
                    meta=Meta(
                        code=500,
                        message="Maaf, KTP tidak terdeteksi",
                        error=True,
                    ),
                )
            ),
        )
