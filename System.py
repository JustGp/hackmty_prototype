#System
#pip install ultralytics
#pip install imutils
#pip install paddlepaddle
#pip install paddleocr

from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import imutils
import re

model = YOLO("hackaton/wine.pt")
ocr = PaddleOCR(use_angle_cls=True , lang='en')

cap = cv2.VideoCapture(0)
while cap.isOpened():


    # Leemos el frame del video
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)

    for result in results:
        #                   clase de el label
        index = (result.boxes.cls == 1).nonzero(as_tuple=True)[0]


        for idx in index:
            conf = result.boxes.conf[idx].item()
            if conf > 0.7:
                xyxy = result.boxes.xyxy[idx].squeeze().tolist()
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                x2, y2 = int(xyxy[2]), int(xyxy[3])

                # Clamp coordinates with padding and make a safe crop
                h, w = frame.shape[:2]
                pad = 15
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(w, x2 + pad)
                y2p = min(h, y2 + pad)
                if x2p <= x1p or y2p <= y1p:
                    continue
                label_image = frame[y1p:y2p, x1p:x2p]

                # Validate crop before OCR
                if label_image is None or label_image.size == 0:
                    continue
                # skip very small patches
                if label_image.shape[0] < 10 or label_image.shape[1] < 10:
                    continue

                # PaddleOCR: attempt to run and parse results
                ocr_img = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
                try:
                    result_ocr = ocr.ocr(ocr_img, cls=True)
                except Exception:
                    try:
                        # fallback signature
                        result_ocr = ocr.ocr(ocr_img)
                    except Exception as e:
                        print("OCR failed:", e)
                        continue

                # Extract texts and confidences, then sort left-to-right
                texts = []  # list of (left_x, text, conf)
                for line in result_ocr:
                    # typical line format: [ (box, (text, confidence)), ... ]
                    if isinstance(line, list):
                        for box_info in line:
                            if not box_info:
                                continue
                            box = box_info[0]
                            if len(box_info) > 1 and isinstance(box_info[1], (list, tuple)):
                                txt = str(box_info[1][0])
                                conf = float(box_info[1][1]) if len(box_info[1]) > 1 else None
                            else:
                                # unexpected format
                                continue
                            left_x = min([pt[0] for pt in box])
                            texts.append((left_x, txt.upper(), conf))
                    else:
                        # try to handle alternate tuple format
                        try:
                            box = line[0]
                            txt = str(line[1][0])
                            conf = float(line[1][1]) if len(line[1]) > 1 else None
                            left_x = min([pt[0] for pt in box])
                            texts.append((left_x, txt.upper(), conf))
                        except Exception:
                            continue

                texts_sorted = [t for _, t in sorted(texts, key=lambda x: x[0])]

                # Print each detected word and its confidence to the terminal
                if texts_sorted:
                    print("Detected OCR tokens:")
                    for tok in texts_sorted:
                        lx, txt, conf = tok
                        if conf is not None:
                            print(f"  '{txt}'  (conf: {conf:.3f})")
                        else:
                            print(f"  '{txt}'")
                else:
                    print("No OCR tokens detected")

                # Build final filtered output (whitelist A-Z0-9)
                whitelist_pattern = re.compile(r'^[A-Z0-9]+$')
                output_text = ''.join([t for _, t, _ in texts_sorted if whitelist_pattern.fullmatch(t)])
                print(f"output_text: {output_text}")
                cv2.imshow("plate_image", label_image)



    print(results[0].boxes)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Inference", annotated_frame)


    


    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
