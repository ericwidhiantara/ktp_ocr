from pydantic import BaseModel
from typing import Optional, Generic, TypeVar

T = TypeVar('T')


class Meta(BaseModel):
    code: int = 200
    message: str = "success"
    error: bool = False


class BaseResp(BaseModel, Generic[T]):
    meta: Meta = Meta()
    data: Optional[T] = None


class AlamatSchema(BaseModel):
    name: str = ""
    rt_rw: str = ""
    kel_desa: str = ""
    kecamatan: str = ""
    kabupaten: str = ""
    provinsi: str = ""


class OcrResultSchema(BaseModel):
    nik: str = ""
    nama: str = ""
    tempat_lahir: str = ""
    tgl_lahir: str = ""
    jenis_kelamin: str = ""
    gol_darah: str = ""
    agama: str = ""
    status_perkawinan: str = ""
    pekerjaan: str = ""
    kewarganegaraan: str = ""
    alamat: AlamatSchema = AlamatSchema()
    time_elapsed: str = ""
