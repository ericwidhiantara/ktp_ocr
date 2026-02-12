import cv2
import numpy as np
import os
import pandas as pd
import pytesseract
import re
import textdistance
import datetime
from operator import itemgetter


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LINE_REC_PATH = os.path.join(ROOT_PATH, 'data/ID_CARD_KEYWORDS.csv')
RELIGION_REC_PATH = os.path.join(ROOT_PATH, 'data/RELIGIONS.csv')
JENIS_KELAMIN_REC_PATH = os.path.join(ROOT_PATH, 'data/JENIS_KELAMIN.csv')
NEED_COLON = [3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21]
NEXT_LINE = 9
ID_NUMBER = 3

# Calibrated ROIs for 1000x600 resolution with ~30px row height
KTP_ROIS = {
    "nik": (237, 120, 450, 45),
    "nama": (245, 148, 550, 55),
    "tempat_tgl_lahir": (260, 178, 550, 55),
    "jenis_kelamin": (260, 208, 300, 55),
    "gol_darah": (600, 208, 150, 55),
    "alamat": (250, 238, 500, 55),
    "rt_rw": (280, 268, 250, 55),
    "kel_desa": (280, 298, 450, 55),
    "kecamatan": (280, 328, 450, 55),
    "agama": (280, 358, 450, 55),
    "status_perkawinan": (280, 388, 450, 55),
    "pekerjaan": (280, 418, 550, 55),
    "kewarganegaraan": (280, 448, 350, 55),
    "provinsi": (200, 10, 600, 60),
    "kabupaten": (250, 50, 500, 60),
}


class OcrService:
    """KTP OCR processing service."""

    @staticmethod
    def _convert_scale(img, alpha, beta):
        new_img = img * alpha + beta
        new_img[new_img < 0] = 0
        new_img[new_img > 255] = 255
        return new_img.astype(np.uint8)

    @staticmethod
    def _automatic_brightness_and_contrast(image, clip_hist_percent=10):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        accumulator = []
        accumulator.append(float(hist[0][0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index][0]))

        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = OcrService._convert_scale(image, alpha=alpha, beta=beta)
        return auto_result

    @staticmethod
    def _sort_contours(cnts, method="left-to-right"):
        reverse = False
        i = 0

        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(
            *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
        )

        return cnts, boundingBoxes

    @staticmethod
    def _return_id_number(image, img_gray):
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, rectKernel)

        gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)

        threshCnts, hierarchy = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = threshCnts
        cur_img = image.copy()
        cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
        copy = image.copy()

        locs = []
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)

            if h > 10 and w > 100 and x < 300:
                img = cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                locs.append((x, y, w, h, w * h))

        locs = sorted(locs, key=itemgetter(1), reverse=False)

        check_nik = False

        try:
            # Refined NIK selection: Look for a horizontal block that isn't the header
            # Usually height 20-70, width 300-700 after 1000x600 resize
            best_nik = None
            for loc in locs:
                x, y, w, h, area = loc
                if 250 < w < 800 and 15 < h < 100:
                    # Prefer the one slightly lower than the very top (to avoid PROVINSI)
                    # but not too low (NIK is usually in the top half)
                    if 80 < y < 270:
                        best_nik = loc
                        break
            
            if not best_nik and locs:
                # Fallback to index 1 or 0
                best_nik = locs[1] if len(locs) > 1 else locs[0]
            
            if not best_nik: return "", None
            
            target = best_nik
            nik = image[
                max(0, target[1] - 15) : min(image.shape[0], target[1] + target[3] + 15),
                max(0, target[0] - 15) : min(image.shape[1], target[0] + target[2] + 15),
            ]
            check_nik = True
            gX, gY, gW, gH = target[0], target[1], target[2], target[3]
        except Exception:
            return "", None

        if check_nik:
            img_mod = cv2.imread(os.path.join(ROOT_PATH, "data/module2.png"))

            ref = cv2.cvtColor(img_mod, cv2.COLOR_BGR2GRAY)
            ref = cv2.threshold(ref, 66, 255, cv2.THRESH_BINARY_INV)[1]

            refCnts, hierarchy = cv2.findContours(
                ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            refCnts = OcrService._sort_contours(refCnts, method="left-to-right")[0]

            digits = {}
            for (i, c) in enumerate(refCnts):
                (x, y, w, h) = cv2.boundingRect(c)
                roi = ref[y : y + h, x : x + w]
                roi = cv2.resize(roi, (57, 88))
                digits[i] = roi

            gray_nik = cv2.cvtColor(nik, cv2.COLOR_BGR2GRAY)
            group = cv2.threshold(gray_nik, 127, 255, cv2.THRESH_BINARY_INV)[1]

            digitCnts, hierarchy_nik = cv2.findContours(
                group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            nik_r = nik.copy()

            ctx = OcrService._sort_contours(digitCnts, method="left-to-right")[0]

            locs_x = []
            for (i, c) in enumerate(ctx):
                (x, y, w, h) = cv2.boundingRect(c)
                if h > 10 and w > 100: # Dummy condition to populate locs_x if needed, but ctx is the source now
                   pass
                if h > 10 and w > 10:
                    locs_x.append((x, y, w, h))

            output = []
            groupOutput = []

            for c in locs_x:
                (x, y, w, h) = c
                roi = group[y : y + h, x : x + w]
                roi = cv2.resize(roi, (57, 88))

                scores = []
                for (digit, digitROI) in digits.items():
                    result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)

                groupOutput.append(str(np.argmax(scores)))

            output.extend(groupOutput)
            return "".join(output), (gX, gY, gW, gH)
        else:
            return "", None

    @staticmethod
    def _ocr_raw(image):
        # Increased resolution for ROI-based extraction
        image = cv2.resize(image, (1000, 600), interpolation=cv2.INTER_CUBIC)

        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

        gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, rectKernel)

        return_id = OcrService._return_id_number(image, blackhat)
        if isinstance(return_id, tuple):
            id_number, nik_box = return_id
        else:
            id_number, nik_box = return_id, None

        if id_number == "":
            raise Exception("KTP tidak terdeteksi")

        # Fill photo area to avoid noise
        cv2.fillPoly(
            blackhat,
            pts=[
                np.asarray(
                    [(650, 150), (650, 599), (998, 599), (998, 150)]
                )
            ],
            color=(255, 255, 255),
        )
        th, threshed = cv2.threshold(blackhat, 130, 255, cv2.THRESH_TRUNC)

        result_raw = pytesseract.image_to_string(
            threshed, lang="ind", config="--psm 6 --oem 3"
        )

        return result_raw, id_number, image, nik_box

    @staticmethod
    def _extract_anchored_roi(image, nik_box, roi_key):
        """Extract a field ROI anchored relative to the NIK position."""
        if nik_box is None:
            return None
            
        # Calibrated NIK reference point from 1.jpg at 1000x600
        ref_nik_x, ref_nik_y = 237, 123
        
        # Current NIK box
        nx, ny, nw, nh = nik_box
        
        # Target ROI coordinates from KTP_ROIS
        rx, ry, rw, rh = KTP_ROIS[roi_key]
        
        # Calculate offset relative to NIK
        shifted_x = rx + (nx - ref_nik_x)
        shifted_y = ry + (ny - ref_nik_y)
        
        # Clamp to image bounds
        h1, w1 = image.shape[:2]
        x1 = max(0, int(shifted_x))
        y1 = max(0, int(shifted_y))
        x2 = min(w1, int(shifted_x + rw))
        y2 = min(h1, int(shifted_y + rh))
        
        roi = image[y1:y2, x1:x2]
        return roi

    @staticmethod
    def _extract_gol_darah(image):
        if image is None:
            return ""

        # Crop to include "Jenis Kelamin" label as anchor
        roi = image[165:215, 300:750]
        
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Use CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Otsu helps separate text from background
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Also try inverted (White text on black background) just in case
            thresh_inv = cv2.bitwise_not(thresh)

            # Helper to try OCR on an image
            def try_ocr(img_input, psm_mode):
                custom_config = f'--oem 3 --psm {psm_mode}'
                return pytesseract.image_to_string(img_input, config=custom_config, lang="ind").strip()

            # Try variants with PSM 6 (Block)
            text_raw = try_ocr(gray, 6)
            text_clahe = try_ocr(enhanced, 6)
            text_thresh = try_ocr(thresh, 6)
            text_inv = try_ocr(thresh_inv, 6)

            # Combine candidates
            candidates = [text_raw, text_clahe, text_thresh, text_inv]
            
            for raw in candidates:
                raw = raw.replace("—", "-").replace("O0", "O").replace("0O", "O")

                # 1. Look for explicit keyword followed by valid value
                match_keyword = re.search(r"(?:GOL|DARAH|DAR|AH|BAMH|BARAH)\s*[:.]?\s*([ABO]{1,2}[+-]?)", raw, re.IGNORECASE)
                if match_keyword:
                    val = match_keyword.group(1).upper().strip()
                    if val in ["A", "B", "AB", "O", "-", "A+", "B+", "O+", "AB+", "A-", "B-", "O-", "AB-"]:
                        return val

                # 2. Look for standalone valid value
                valid_types = r"\b(A|B|AB|O)[+-]?\b"
                matches = re.finditer(valid_types, raw)
                
                for m in matches:
                    val = m.group(0).upper().strip()
                    if val in ["A", "B", "AB", "O", "-", "A+", "B+", "O+", "AB+", "A-", "B-", "O-", "AB-"]:
                         return val
                
                # 3. Handle common misreads (0->O, 8->B)
                raw_clean = raw.replace("0", "O").replace("8", "B")
                matches = re.finditer(valid_types, raw_clean)
                for m in matches:
                    val = m.group(0).upper().strip()
                    if val in ["A", "B", "AB", "O", "-", "A+", "B+", "O+", "AB+", "A-", "B-", "O-", "AB-"]:
                         return val

                # 4. Check for '-'
                if "-" in raw:
                      if re.search(r"(?:GOL|DARAH|DAR|AH).*[-]", raw, re.IGNORECASE):
                         return "-"
                      match_dash = re.search(r"\b-\b", raw)
                      if match_dash:
                          return "-"

        except Exception:
            return ""

        return ""

    @staticmethod
    def _preprocess_ocr_text(text):
        """Pre-process raw OCR text to fix common Tesseract merge issues."""
        replacements = {
            "JenisKelamin": "Jenis Kelamin",
            "Jeniskelamin": "Jenis Kelamin",
            "jeniskelamin": "Jenis Kelamin",
            "JenisKeiamin": "Jenis Kelamin",
            "Golbarah": "Gol. Darah",
            "GolDarah": "Gol. Darah",
            "GOLDARAH": "Gol. Darah",
            "Gol.Darah": "Gol. Darah",
            "RTRW": "RT/RW",
            "RT.RW": "RT/RW",
            "Kel/desa": "Kel/Desa",
            "KeVDesa": "Kel/Desa",
            "Kei/Desa": "Kel/Desa",
            "Tempai/Tgi": "Tempat/Tgl",
            "Tempat/Tgi": "Tempat/Tgl",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def _strip_op(result_raw):
        result_list = result_raw.split("\n")
        new_result_list = []

        for tmp_result in result_list:
            if tmp_result.strip(" "):
                new_result_list.append(tmp_result)

        return new_result_list

    @staticmethod
    def process(image) -> tuple:
        """
        Process a KTP image using Anchored ROI extraction with hybrid Keyword fallback.
        """
        # 1. Initial OCR to get NIK and Anchor Box
        result_raw, id_number, resized_image, nik_box = OcrService._ocr_raw(image)
        
        # Initialize results
        data = {
            "nik": id_number,
            "nama": "",
            "tempat_lahir": "",
            "tgl_lahir": "",
            "jenis_kelamin": "",
            "gol_darah": "",
            "alamat": "",
            "rt_rw": "",
            "kel_desa": "",
            "kecamatan": "",
            "agama": "",
            "status_perkawinan": "",
            "pekerjaan": "",
            "kewarganegaraan": "",
            "provinsi": "",
            "kabupaten": ""
        }

        # Helper for ROI-based OCR
        def ocr_roi(roi_key, psm=7, cleaning_regex=None):
            roi = OcrService._extract_anchored_roi(resized_image, nik_box, roi_key)
            if roi is None or roi.size == 0:
                return ""
            
            # Preprocessing for better OCR on small crops
            roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Median blur to remove salt-and-pepper noise
            gray = cv2.medianBlur(gray, 3)
            
            # Binarization
            _, enhanced = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR
            config = f"--oem 3 --psm {psm}"
            text = pytesseract.image_to_string(enhanced, lang="ind", config=config).strip()
            
            # Remove common KTP prefix noise
            text = re.sub(r"^(Nama|Alamat|RT/RW|Kel/Desa|Kel.Desa|Kecamatan|Agama|Status|Pekerjaan|NIK|Tempat/Tgl Lahir|Jenis Kelamin|Gol. Darah|Provinsi|Kabupaten|Kota|Kewarganegaraan)\s*[:.]?\s*", "", text, flags=re.IGNORECASE)
            
            # Cleaning
            if cleaning_regex:
                text = "".join(re.findall(cleaning_regex, text))
            
            return text.strip().upper()

        # 2. Primary Extraction via full-page OCR keywords
        lines = result_raw.split("\n")
        
        def find_value(keywords, fallback_roi=None, cleaning_regex=None):
            # Try keywords in full-page OCR first
            for line in lines:
                for kw in keywords:
                    if kw.upper() in line.upper():
                        # Extract everything after the keyword and potential colon/noise
                        # Improved regex to strip common leading punctuation/markers
                        val = re.sub(rf"^.*?{kw}\s*[:.;, \-—'“]*", "", line, flags=re.IGNORECASE).strip()
                        if val:
                            if cleaning_regex:
                                val = "".join(re.findall(cleaning_regex, val))
                            return val.strip().upper()
            
            # Fallback to targeted ROI if keyword not found
            if fallback_roi:
                return ocr_roi(fallback_roi, cleaning_regex=cleaning_regex)
            return ""

        data["nama"] = find_value(["Nama"], fallback_roi="nama", cleaning_regex="[A-Z. ]")
        
        ttl_raw = find_value(["Tempat/Tgl Lahir", "Tempat", "Lahir"], fallback_roi="tempat_tgl_lahir")
        if ttl_raw:
             # Strip "GILAHIR", "TGL", "TEMPAT" etc. that might be read as part of the value
             ttl_raw = re.sub(r"^(GILAHIR|TGL|TEMPAT|LAHIR|TEMPAT/TGL LAHIR|TEMPAT/TGL|TGL LAHIR)\s*[:.;, \-—]*", "", ttl_raw, flags=re.IGNORECASE).strip()
             
             match_tgl = re.search(r"(\d{2}[- ]\d{2}[- ]\d{4})", ttl_raw)
             if match_tgl:
                 data["tgl_lahir"] = match_tgl.group(1).replace(" ", "-")
                 data["tempat_lahir"] = "".join(re.findall(r"[A-Z ]", ttl_raw[:match_tgl.start()])).strip()
             else:
                 data["tempat_lahir"] = "".join(re.findall(r"[A-Z ]", ttl_raw)).strip()
        
        data["jenis_kelamin"] = find_value(["Jenis Kelamin", "Kelamin"], fallback_roi="jenis_kelamin")
        if "LAKI" in data["jenis_kelamin"]: data["jenis_kelamin"] = "LAKI-LAKI"
        elif "PEREMPUAN" in data["jenis_kelamin"]: data["jenis_kelamin"] = "PEREMPUAN"

        data["gol_darah"] = find_value(["Gol. Darah", "Darah"], fallback_roi="gol_darah")
        # Strict cleaning for Blood Type to avoid "€" etc.
        gol_match = re.search(r"\b([ABO]{1,2}[+-]?)\b", data["gol_darah"])
        data["gol_darah"] = gol_match.group(1) if gol_match else ""

        data["alamat"] = find_value(["Alamat"], fallback_roi="alamat", cleaning_regex="[A-Z0-9. /,-]")
        
        # Address Space Restoration (e.g. PRMPURI -> PRM PURI)
        common_prefixes = ["PRM", "PURI", "BLOK", "NO", "DS", "DSN", "JL", "JALAN"]
        for pref in common_prefixes:
            # If prefix is merged with next word (Uppercase), add space
            data["alamat"] = re.sub(rf"\b({pref})([A-Z0-9])", r"\1 \2", data["alamat"])
        
        # Strip trailing noise/punctuation
        data["alamat"] = data["alamat"].strip(". -," )
        
        # Format normalization: dots are often misread commas in addresses
        data["alamat"] = data["alamat"].replace(".", ",")
            
        data["rt_rw"] = find_value(["RT/RW"], fallback_roi="rt_rw", cleaning_regex="[0-9/]")
        data["kel_desa"] = find_value(["Kel/Desa", "Desa"], fallback_roi="kel_desa", cleaning_regex="[A-Z0-9. /,-]")
        data["kecamatan"] = find_value(["Kecamatan"], fallback_roi="kecamatan", cleaning_regex="[A-Z0-9. /,-]")
        
        religion_df = pd.read_csv(RELIGION_REC_PATH, header=None)
        agama_raw = find_value(["Agama"], fallback_roi="agama")
        if agama_raw:
            # Misread fix: "SBAM" -> "ISLAM"
            if "SBAM" in agama_raw: agama_raw = "ISLAM"
            
            sims = [textdistance.damerau_levenshtein.normalized_similarity(agama_raw, r) for r in religion_df[0].values]
            if max(sims) >= 0.6:
                data["agama"] = religion_df[0].values[np.argmax(sims)]
            else:
                data["agama"] = agama_raw

        # Status matching
        status_raw = find_value(["Status Perkawinan", "Status", "Slaltus", "Staltus"], fallback_roi="status_perkawinan")
        if "BELUM KAWIN" in status_raw: data["status_perkawinan"] = "BELUM KAWIN"
        elif "CERAI HIDUP" in status_raw: data["status_perkawinan"] = "CERAI HIDUP"
        elif "CERAI MATI" in status_raw: data["status_perkawinan"] = "CERAI MATI"
        elif "KAWIN" in status_raw: data["status_perkawinan"] = "KAWIN"
        else: data["status_perkawinan"] = status_raw

        data["pekerjaan"] = find_value(["Pekerjaan"], fallback_roi="pekerjaan", cleaning_regex="[A-Z. /]")
        
        kw_raw = find_value(["Kewarganegaraan"], fallback_roi="kewarganegaraan")
        if "WNI" in kw_raw: data["kewarganegaraan"] = "WNI"
        elif "WNA" in kw_raw: data["kewarganegaraan"] = "WNA"
        else: data["kewarganegaraan"] = kw_raw

        # 3. Validation/Fallbacks
        if not data["tgl_lahir"] and id_number and len(id_number) == 16 and id_number.isdigit():
            try:
                date_part = id_number[6:12]
                dd = int(date_part[:2])
                if dd > 40: dd -= 40
                mm = int(date_part[2:4])
                yy = int(date_part[4:6])
                curr_year = datetime.datetime.now().year % 100
                year_full = 2000 + yy if yy < curr_year + 5 else 1900 + yy
                data["tgl_lahir"] = f"{dd:02d}-{mm:02d}-{year_full}"
            except Exception: pass

        if id_number and len(id_number) == 16:
            try:
                from app.services.nik_service import NikService
                prov, kab, kec = NikService.parse_nik(id_number)
                data["provinsi"] = prov if prov else data["provinsi"]
                data["kabupaten"] = kab if kab else data["kabupaten"]
                if not data["kecamatan"]: data["kecamatan"] = kec
            except Exception: pass

        if not data["gol_darah"]:
            data["gol_darah"] = OcrService._extract_gol_darah(resized_image)

        return (
            data["nik"], data["nama"], data["tempat_lahir"], data["tgl_lahir"],
            data["jenis_kelamin"], data["gol_darah"], data["agama"],
            data["status_perkawinan"], data["provinsi"], data["kabupaten"],
            data["alamat"], data["rt_rw"], data["kel_desa"],
            data["kecamatan"], data["pekerjaan"], data["kewarganegaraan"]
        )
