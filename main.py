
"""
main.py
AI for Accessibility - Mini Project
Updated to YOLOv8 (Ultralytics)
"""
import argparse
import time
from pathlib import Path
import cv2
import torch
import pyttsx3
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# ----------------------------# CONFIG# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTION_MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
TTS_ENABLED = True

DETECT_CONF_THRESH = 0.35
OBJECT_SIZE_CLOSE_THRESHOLD = 0.25
OBJECT_SIZE_NEAR_THRESHOLD = 0.06

# ----------------------------# TTS# ----------------------------
class TTS:
    def __init__(self, enabled=True):
        self.enabled = enabled
        if not enabled:
            return 0
            
        try:
            self.engine = pyttsx3.init()
            rate = self.engine.getProperty("rate")
            self.engine.setProperty("rate", int(rate * 0.9))
        except Exception as e:
            print("TTS init failed:", e)
            self.enabled = False

    def say(self, text):
        if not self.enabled:
            print("[TTS disabled] >", text)
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except:
            self.enabled = False

# ----------------------------# Captioning# ----------------------------
class ImageCaptioner:
    def __init__(self, model_name=CAPTION_MODEL_NAME, device=DEVICE):
        print("Loading captioning model... (may download weights first run)")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    def caption(self, pil_image):
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        pixel_values = self.processor(images=pil_image,
                                      return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(pixel_values,
                                             max_length=64,
                                             num_beams=4)

        result = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return result[0].strip()

# ----------------------------# Navigation# ----------------------------
def compute_area_fraction(box, frame_shape):
    x1, y1, x2, y2 = box
    frame_h, frame_w = frame_shape[:2]
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_h * frame_w
    return box_area / (frame_area + 1e-9)

def navigation_cue(detections, frame_shape):
    if not detections:
        return "No objects detected. Area looks clear."

    largest = max(detections, key=lambda d: compute_area_fraction(d["box"], frame_shape))
    frac = compute_area_fraction(largest["box"], frame_shape)
    label = largest["label"]

    x1, y1, x2, y2 = largest["box"]
    frame_w = frame_shape[1]
    cx = (x1 + x2) / 2

    position = "center"
    if cx < frame_w * 0.35:
        position = "left"
    elif cx > frame_w * 0.65:
        position = "right"

    if frac >= OBJECT_SIZE_CLOSE_THRESHOLD:
        return f"{label} very close at your {position}. Stop and step back."
    elif frac >= OBJECT_SIZE_NEAR_THRESHOLD:
        return f"{label} near at your {position}. Move slightly to your opposite side."
    else:
        return f"{label} far at your {position}. You can move forward."

# ----------------------------# YOLOv8# ----------------------------
def load_yolo_model():
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")   # tiny, fast
    return model

# ----------------------------# Realtime webcam# ----------------------------
def run_realtime(captioner, tts, yolo):
    print("Starting webcam... press 'q' to exit")
    cap = cv2.VideoCapture(0)
    last_caption_time = 0
    last_spoken = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 detection
        results = yolo(frame)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = results.names[cls]

            if conf < DETECT_CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            detections.append({"label": label, "conf": conf, "box": (x1, y1, x2, y2)})

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # caption every 3 seconds
        caption_text = ""
        if time.time() - last_caption_time > 3:
            try:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                caption_text = captioner.caption(pil)
            except:
                caption_text = ""
            last_caption_time = time.time()

        nav_text = navigation_cue(detections, frame.shape)

        # overlay
        y = 30
        if caption_text:
            cv2.putText(frame, "Scene: " + caption_text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 30

        cv2.putText(frame, "Navigation: " + nav_text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("AI Accessibility", frame)

        # speak
        speak_text = (caption_text + ". " + nav_text).strip()
        if speak_text and speak_text != last_spoken:
            tts.say(speak_text)
            last_spoken = speak_text

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------# Single Image Mode# ----------------------------
def run_single_image(captioner, tts, yolo, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Cannot open image:", img_path)
        return

    results = yolo(img)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = results.names[cls]

        if conf < DETECT_CONF_THRESH:
            continue

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        detections.append({"label": label, "conf": conf, "box": (x1, y1, x2, y2)})

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    caption_text = captioner.caption(pil)
    nav_text = navigation_cue(detections, img.shape)

    tts.say(caption_text + ". " + nav_text)
    print("Caption:", caption_text)
    print("Navigation:", nav_text)

# ----------------------------# MAIN# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--no-tts", action="store_true")
    args = parser.parse_args()

    tts = TTS(enabled=not args.no_tts and TTS_ENABLED)
    captioner = ImageCaptioner()
    yolo = load_yolo_model()

    if args.image:
        run_single_image(captioner, tts, yolo, args.image)
    else:
        run_realtime(captioner, tts, yolo)

if __name__ == "__main__":
    main()