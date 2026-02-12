import json
import os
from typing import Dict, Optional, Tuple

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WILAYAH_PATH = os.path.join(ROOT_PATH, 'data/wilayah.json')

class NikService:
    _wilayah_data = None

    @classmethod
    def _load_data(cls):
        if cls._wilayah_data is None:
            if os.path.exists(WILAYAH_PATH):
                with open(WILAYAH_PATH, 'r') as f:
                    cls._wilayah_data = json.load(f)
            else:
                print(f"Warning: wilayah.json not found at {WILAYAH_PATH}")
                cls._wilayah_data = {}

    @classmethod
    def parse_nik(cls, nik: str) -> Tuple[str, str, str]:
        """
        Parse NIK and return (Provinsi, Kabupaten/Kota, Kecamatan).
        Returns empty strings if not found or NIK is invalid.
        """
        cls._load_data()
        
        if not nik or len(nik) < 6:
            return "", "", ""

        prov_id = nik[:2]
        kab_id = nik[2:4]
        kec_id = nik[4:6]

        prov_name = ""
        kab_name = ""
        kec_name = ""

        # Lookup Provinsi
        if "provinsi" in cls._wilayah_data and prov_id in cls._wilayah_data["provinsi"]:
            prov_name = cls._wilayah_data["provinsi"][prov_id]

        # Lookup Kabupaten
        if "kabupaten" in cls._wilayah_data and prov_id in cls._wilayah_data["kabupaten"]:
            if kab_id in cls._wilayah_data["kabupaten"][prov_id]:
                kab_name = cls._wilayah_data["kabupaten"][prov_id][kab_id]

        # Lookup Kecamatan
        # Key format in JSON seems to be "ProvIdKabId" (4 digits)
        kec_key = prov_id + kab_id
        if "kecamatan" in cls._wilayah_data and kec_key in cls._wilayah_data["kecamatan"]:
            if kec_id in cls._wilayah_data["kecamatan"][kec_key]:
                kec_name = cls._wilayah_data["kecamatan"][kec_key][kec_id]

        return prov_name, kab_name, kec_name
