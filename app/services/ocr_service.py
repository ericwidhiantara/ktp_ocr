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
            nik = image[
                locs[1][1] - 15 : locs[1][1] + locs[1][3] + 15,
                locs[1][0] - 15 : locs[1][0] + locs[1][2] + 15,
            ]
            check_nik = True
        except Exception as e:
            print(e)
            return ""

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
            cv2.drawContours(nik_r, digitCnts, -1, (0, 0, 255), 3)

            gX = locs[1][0]
            gY = locs[1][1]
            gW = locs[1][2]
            gH = locs[1][3]

            ctx = OcrService._sort_contours(digitCnts, method="left-to-right")[0]

            locs_x = []
            for (i, c) in enumerate(ctx):
                (x, y, w, h) = cv2.boundingRect(c)
                if h > 10 and w > 10:
                    img = cv2.rectangle(nik_r, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

            cv2.rectangle(
                image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1
            )
            cv2.putText(
                image,
                "".join(groupOutput),
                (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            output.extend(groupOutput)
            return "".join(output)
        else:
            return ""

    @staticmethod
    def _ocr_raw(image):
        image = cv2.resize(image, (50 * 16, 500))

        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

        gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, rectKernel)

        id_number = OcrService._return_id_number(image, blackhat)
        if id_number == "":
            raise Exception("KTP tidak terdeteksi")

        cv2.fillPoly(
            blackhat,
            pts=[
                np.asarray(
                    [(550, 150), (550, 499), (798, 499), (798, 150)]
                )
            ],
            color=(255, 255, 255),
        )
        th, threshed = cv2.threshold(blackhat, 130, 255, cv2.THRESH_TRUNC)

        result_raw = pytesseract.image_to_string(
            threshed, lang="ind", config="--psm 4 --oem 3"
        )

        print(result_raw)

        return result_raw, id_number, image

    @staticmethod
    def _extract_gol_darah(image):
        if image is None:
            return ""

        # Crop to include "Jenis Kelamin" label as anchor
        # y: 165-215 (approx 50px height centering on the row)
        # x: 300-750 (include "Jenis Kelamin" from left, avoid photo edge on right)
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
                custom_config = f'--oem 3 --psm {psm_mode}' # No whitelist to see raw output
                return pytesseract.image_to_string(img_input, config=custom_config, lang="ind").strip()

            # Try variants with PSM 6 (Block)
            # 1. Raw Grayscale (let Tesseract handle binarization)
            text_raw = try_ocr(gray, 6)
            # 2. CLAHE Enhanced
            text_clahe = try_ocr(enhanced, 6)
            # 3. Normal Thresh
            text_thresh = try_ocr(thresh, 6)
            # 4. Inverted Thresh
            text_inv = try_ocr(thresh_inv, 6)

            # Combine candidates
            candidates = [text_raw, text_clahe, text_thresh, text_inv]
            
            # Regex to find blood type
            # We look for "Gol. Darah" or "LAKI" context if possible, but mainly the value
            
            for raw in candidates:
                # Clean up specific common misreads in raw text
                # e.g. "Golbamh" -> "Gol. Darah" to help regex, or just ignore labels
                # "O-" often read as "O—" (em dash) or "O-—"
                raw = raw.replace("—", "-").replace("O0", "O").replace("0O", "O")

                # 1. Look for explicit keyword followed by valid value
                # Keywords: GOL, DARAH, Golbamh, Golbarah, Gol.Darah
                match_keyword = re.search(r"(?:GOL|DARAH|DAR|AH|BAMH|BARAH)\s*[:.]?\s*([ABO]{1,2}[+-]?)", raw, re.IGNORECASE)
                if match_keyword:
                    val = match_keyword.group(1).upper().strip()
                    if val in ["A", "B", "AB", "O", "-", "A+", "B+", "O+", "AB+", "A-", "B-", "O-", "AB-"]:
                        return val

                # 2. Look for standalone valid value with strict word boundaries
                # Must be A, B, AB, O, or -
                # We iterate all matches and pick the one that is NOT a substring of a keyword or gender
                
                # Regex for valid blood types only
                valid_types = r"\b(A|B|AB|O)[+-]?\b"
                matches = re.finditer(valid_types, raw)
                
                for m in matches:
                    val = m.group(0).upper().strip()
                    # Extra safety: Ensure it's in our strict list
                    if val in ["A", "B", "AB", "O", "-", "A+", "B+", "O+", "AB+", "A-", "B-", "O-", "AB-"]:
                         return val
                
                # 3. Handle common misreads (0->O, 8->B) with boundaries
                raw_clean = raw.replace("0", "O").replace("8", "B")
                matches = re.finditer(valid_types, raw_clean)
                for m in matches:
                    val = m.group(0).upper().strip()
                    if val in ["A", "B", "AB", "O", "-", "A+", "B+", "O+", "AB+", "A-", "B-", "O-", "AB-"]:
                         return val

                # 4. Check for '-'
                if "-" in raw:
                     # If specifically 'Gol. Darah -' or similar
                     if re.search(r"(?:GOL|DARAH|DAR|AH).*[-]", raw, re.IGNORECASE):
                        return "-"
                     # If it's just a dash at the end/start?
                     match_dash = re.search(r"\b-\b", raw)
                     if match_dash:
                         return "-"

        except Exception as e:
            print(f"[OCR Debug] Gol. Darah extraction failed: {str(e)}")
            return ""

        return ""

    @staticmethod
    def _preprocess_ocr_text(text):
        """Pre-process raw OCR text to fix common Tesseract merge issues."""
        # Split commonly merged keywords
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
        Process a KTP image and extract OCR data.

        Args:
            image: OpenCV image (numpy array)

        Returns:
            Tuple of (nik, nama, tempat_lahir, tgl_lahir, jenis_kelamin,
                      agama, status_perkawinan, provinsi, kabupaten, alamat,
                      rt_rw, kel_desa, kecamatan, pekerjaan, kewarganegaraan)
        """
        raw_df = pd.read_csv(LINE_REC_PATH, header=None)
        religion_df = pd.read_csv(RELIGION_REC_PATH, header=None)
        jenis_kelamin_df = pd.read_csv(JENIS_KELAMIN_REC_PATH, header=None)
        result_raw, id_number, resized_image = OcrService._ocr_raw(image)
        result_raw = OcrService._preprocess_ocr_text(result_raw)
        result_list = OcrService._strip_op(result_raw)

        provinsi = ""
        kabupaten = ""
        nik = ""
        nama = ""
        tempat_lahir = ""
        tgl_lahir = ""
        jenis_kelamin = ""
        gol_darah = ""
        alamat = ""
        status_perkawinan = ""
        agama = ""
        rt_rw = ""
        kel_desa = ""
        kecamatan = ""
        pekerjaan = ""
        kewarganegaraan = ""

        loc2index = dict()
        for i, tmp_line in enumerate(result_list):
            for j, tmp_word in enumerate(tmp_line.split(" ")):
                tmp_sim_list = [
                    textdistance.damerau_levenshtein.normalized_similarity(
                        tmp_word_, tmp_word.strip(":")
                    )
                    for tmp_word_ in raw_df[0].values
                ]

                tmp_sim_np = np.asarray(tmp_sim_list)
                arg_max = np.argmax(tmp_sim_np)

                if tmp_sim_np[arg_max] >= 0.6:
                    loc2index[(i, j)] = arg_max

        last_result_list = []
        useful_info = False

        for i, tmp_line in enumerate(result_list):
            tmp_list = []
            for j, tmp_word in enumerate(tmp_line.split(" ")):
                tmp_word = tmp_word.strip(":")

                if (i, j) in loc2index:
                    useful_info = True
                    if loc2index[(i, j)] == NEXT_LINE:
                        last_result_list.append(tmp_list)
                        tmp_list = []
                    tmp_list.append(raw_df[0].values[loc2index[(i, j)]])
                    if loc2index[(i, j)] in NEED_COLON:
                        tmp_list.append(":")
                elif tmp_word == ":" or tmp_word == "":
                    continue
                else:
                    tmp_list.append(tmp_word)

            if useful_info:
                if len(last_result_list) > 2 and ":" not in tmp_list:
                    last_result_list[-1].extend(tmp_list)
                else:
                    last_result_list.append(tmp_list)

        for tmp_data in last_result_list:
            if "—" in tmp_data:
                tmp_data.remove("—")

            if "PROVINSI" in tmp_data:
                provinsi = " ".join(tmp_data[1:])
                provinsi = re.sub("[^A-Z. ]", "", provinsi).strip()
                # Remove "PROVINSI" prefix if accidentally included
                provinsi = re.sub(r"^PROVINSI\s*", "", provinsi).strip()

                if len(provinsi.split()) == 1:
                    provinsi = re.sub("[^A-Z.]", "", provinsi)

            if "KABUPATEN" in tmp_data or "KOTA" in tmp_data:
                kabupaten = " ".join(tmp_data[1:])
                kabupaten = re.sub("[^A-Z. ]", "", kabupaten).strip()
                # Remove "KABUPATEN"/"KOTA" prefix if accidentally included
                kabupaten = re.sub(r"^(KABUPATEN|KOTA)\s*", "", kabupaten).strip()

                if len(kabupaten.split()) == 1:
                    kabupaten = re.sub("[^A-Z.]", "", kabupaten)

            if "Nama" in tmp_data:
                nama = " ".join(tmp_data[2:])
                nama = re.sub("[^A-Z. ]", "", nama)
                # Strip leading/trailing dots and spaces
                nama = nama.strip(". ").strip()

                if len(nama.split()) == 1:
                    nama = re.sub("[^A-Z.]", "", nama)

            if "NIK" in tmp_data:
                if len(id_number) != 16:
                    if "D" in id_number:
                        id_number = id_number.replace("D", "0")
                    if "?" in id_number:
                        id_number = id_number.replace("?", "7")
                    if "L" in id_number:
                        id_number = id_number.replace("L", "1")

                    while len(tmp_data) > 2:
                        tmp_data.pop()
                    tmp_data.append(id_number)
                else:
                    while len(tmp_data) > 3:
                        tmp_data.pop()
                    if len(tmp_data) < 3:
                        tmp_data.append(id_number)
                    tmp_data[2] = id_number

            if "Agama" in tmp_data:
                for tmp_index, tmp_word in enumerate(tmp_data[1:]):
                    tmp_sim_list = [
                        textdistance.damerau_levenshtein.normalized_similarity(
                            tmp_word, tmp_word_
                        )
                        for tmp_word_ in religion_df[0].values
                    ]

                    tmp_sim_np = np.asarray(tmp_sim_list)
                    arg_max = np.argmax(tmp_sim_np)

                    print(tmp_sim_np[arg_max])

                    if tmp_sim_np[arg_max] >= 0.6:
                        tmp_data[tmp_index + 1] = religion_df[0].values[arg_max]
                        agama = tmp_data[tmp_index + 1]

            if "Status" in tmp_data or "Perkawinan" in tmp_data:
                try:
                    status_perkawinan = " ".join(tmp_data[2:])
                    status_perkawinan = re.findall(
                        "\\s+([A-Za-z]+)", status_perkawinan
                    )
                    status_perkawinan = " ".join(status_perkawinan)
                except:
                    status_perkawinan = ""

            if "Alamat" in tmp_data:
                for tmp_index in range(len(tmp_data)):
                    if "!" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("!", "I")
                    if "1" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("1", "I")
                    if "i" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("i", "I")
                    alamat = " ".join(tmp_data[1:])
                    alamat = re.sub("[^A-Z0-9. ]", "", alamat).strip()

                    if len(alamat.split()) == 1:
                        alamat = re.sub("[^A-Z0-9.]", "", alamat).strip()

            if "RT/RW" in tmp_data:
                for tmp_index in range(len(tmp_data)):
                    if "!" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("!", "1")
                    if "i" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("i", "1")
                    rt_rw = " ".join(tmp_data[1:])
                    # Try standard ###/### pattern first
                    rt_rw_match = re.search(r"\d{3}/\d{3}", rt_rw)
                    if rt_rw_match:
                        rt_rw = rt_rw_match.group()
                    else:
                        # Handle OCR misread where `/` is read as a digit
                        rt_rw_digits = re.search(r"(\d{6,7})", rt_rw)
                        if rt_rw_digits:
                            digits = rt_rw_digits.group()
                            if len(digits) == 7:
                                # 7 digits: middle digit is OCR noise for `/` (e.g. 0017024 -> 001/024)
                                rt_rw = digits[:3] + "/" + digits[4:7]
                            else:
                                # 6 digits: split evenly (e.g. 001024 -> 001/024)
                                rt_rw = digits[:3] + "/" + digits[3:6]
                        else:
                            rt_rw = re.sub(r"[^0-9/]", "", rt_rw).strip()

            if "Kel/Desa" in tmp_data:
                for tmp_index in range(len(tmp_data)):
                    if "!" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("!", "I")
                    if "1" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("1", "I")
                    if "i" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("i", "I")
                    kel_desa = " ".join(tmp_data[1:])
                    kel_desa = re.sub("[^A-Z0-9. ]", "", kel_desa).strip()

                    if len(kel_desa.split()) == 1:
                        kel_desa = re.sub("[^A-Z0-9.]", "", kel_desa).strip()

            if "Kecamatan" in tmp_data:
                for tmp_index in range(len(tmp_data)):
                    if "!" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("!", "I")
                    if "1" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("1", "I")
                    if "i" in tmp_data[tmp_index]:
                        tmp_data[tmp_index] = tmp_data[tmp_index].replace("i", "I")
                    kecamatan = " ".join(tmp_data[1:])
                    kecamatan = re.sub("[^A-Z0-9. ]", "", kecamatan).strip()
                    # Remove trailing single character junk (e.g. "NGEMPLAK S")
                    kecamatan = re.sub(r"\s+[A-Z]$", "", kecamatan).strip()

                    if len(kecamatan.split()) == 1:
                        kecamatan = re.sub("[^A-Z0-9.]", "", kecamatan).strip()

            if "Jenis" in tmp_data or "Kelamin" in tmp_data:
                for tmp_index, tmp_word in enumerate(tmp_data[2:]):
                    tmp_sim_list = [
                        textdistance.damerau_levenshtein.normalized_similarity(
                            tmp_word, tmp_word_
                        )
                        for tmp_word_ in jenis_kelamin_df[0].values
                    ]

                    tmp_sim_np = np.asarray(tmp_sim_list)
                    arg_max = np.argmax(tmp_sim_np)

                    if tmp_sim_np[arg_max] >= 0.6:
                        tmp_data[tmp_index + 2] = jenis_kelamin_df[0].values[arg_max]
                        jenis_kelamin = tmp_data[tmp_index + 2]

                # Fallback: check raw joined text for LAKI pattern
                if not jenis_kelamin:
                    joined = " ".join(tmp_data).upper()
                    if "LAKI" in joined:
                        jenis_kelamin = "LAKI-LAKI"
                    elif "PEREMPUAN" in joined or "PREMPUAN" in joined:
                        jenis_kelamin = "PEREMPUAN"

            if "Gol." in tmp_data or "Darah" in tmp_data:
                joined = " ".join(tmp_data)
                gol_match = re.search(r"([ABO]{1,2}[+-]?)", joined)
                if gol_match:
                    gol_darah = gol_match.group().strip()

            if "Pekerjaan" in tmp_data:
                pekerjaan = " ".join(tmp_data[2:])
                pekerjaan = re.sub("[^A-Za-z./ ]", "", pekerjaan)

                if len(pekerjaan.split()) == 1:
                    pekerjaan = re.sub("[^A-Za-z./]", "", pekerjaan)

            if "Kewarganegaraan" in tmp_data:
                kewarganegaraan = " ".join(tmp_data[2:])
                kewarganegaraan = re.sub("[^A-Z. ]", "", kewarganegaraan)

                if len(kewarganegaraan.split()) == 1:
                    kewarganegaraan = re.sub("[^A-Z.]", "", kewarganegaraan)

            if "Tempat" in tmp_data or "Tgl" in tmp_data or "Lahir" in tmp_data:
                join_tmp = " ".join(tmp_data)

                match_tgl1 = re.search(
                    "([0-9]{2}—[0-9]{2}—[0-9]{4})", join_tmp
                )
                match_tgl2 = re.search(
                    "([0-9]{2}\\ [0-9]{2}\\ [0-9]{4})", join_tmp
                )
                match_tgl3 = re.search(
                    "([0-9]{2}\\-[0-9]{2}\\ [0-9]{4})", join_tmp
                )
                match_tgl4 = re.search(
                    "([0-9]{2}\\ [0-9]{2}\\-[0-9]{4})", join_tmp
                )
                match_tgl5 = re.search(
                    "([0-9]{2}-[0-9]{2}-[0-9]{4})", join_tmp
                )
                match_tgl6 = re.search(
                    "([0-9]{2}\\-[0-9]{2}\\-[0-9]{4})", join_tmp
                )

                if match_tgl1:
                    try:
                        tgl_lahir = datetime.datetime.strptime(
                            match_tgl1.group(), "%d—%m—%Y"
                        ).date()
                        tgl_lahir = tgl_lahir.strftime("%d-%m-%Y")
                    except:
                        tgl_lahir = ""
                elif match_tgl2:
                    try:
                        tgl_lahir = datetime.datetime.strptime(
                            match_tgl2.group(), "%d %m %Y"
                        ).date()
                        tgl_lahir = tgl_lahir.strftime("%d-%m-%Y")
                    except:
                        tgl_lahir = ""
                elif match_tgl3:
                    try:
                        tgl_lahir = datetime.datetime.strptime(
                            match_tgl3.group(), "%d-%m %Y"
                        ).date()
                        tgl_lahir = tgl_lahir.strftime("%d-%m-%Y")
                    except:
                        tgl_lahir = ""
                elif match_tgl4:
                    try:
                        tgl_lahir = datetime.datetime.strptime(
                            match_tgl4.group(), "%d %m-%Y"
                        ).date()
                        tgl_lahir = tgl_lahir.strftime("%d-%m-%Y")
                    except:
                        tgl_lahir = ""
                elif match_tgl5:
                    try:
                        tgl_lahir = datetime.datetime.strptime(
                            match_tgl5.group(), "%d-%m-%Y"
                        ).date()
                        tgl_lahir = tgl_lahir.strftime("%d-%m-%Y")
                    except:
                        tgl_lahir = ""
                elif match_tgl6:
                    try:
                        tgl_lahir = datetime.datetime.strptime(
                            match_tgl6.group(), "%d-%m-%Y"
                        ).date()
                        tgl_lahir = tgl_lahir.strftime("%d-%m-%Y")
                    except:
                        tgl_lahir = ""
                else:
                    tgl_lahir = ""

                try:
                    tempat_lahir = " ".join(tmp_data[2:])
                    # Extract only uppercase letters and spaces
                    tempat_lahir = re.findall("[A-Z\\s]", tempat_lahir)
                    tempat_lahir = "".join(tempat_lahir).strip()
                    # Remove trailing junk from other fields bleeding in
                    # (e.g. "GROBOGAN  JK LAKITAKI   G" -> "GROBOGAN")
                    # Stop at known field keywords
                    for kw in ["JK", "LAKI", "PEREMPUAN", "GOL"]:
                        idx = tempat_lahir.find(kw)
                        if idx > 0:
                            tempat_lahir = tempat_lahir[:idx].strip()
                            break
                except:
                    tempat_lahir = ""

        # Fallback: extract blood type via separate OCR pass on right side of KTP
        if not gol_darah:
            gol_darah = OcrService._extract_gol_darah(resized_image)

        return (
            id_number,
            nama,
            tempat_lahir,
            tgl_lahir,
            jenis_kelamin,
            gol_darah,
            agama,
            status_perkawinan,
            provinsi,
            kabupaten,
            alamat,
            rt_rw,
            kel_desa,
            kecamatan,
            pekerjaan,
            kewarganegaraan,
        )




