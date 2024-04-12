import argparse
import torch
import re
from PIL import Image
from time import time
import cv2
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()
LATEST_REVISION = "2024-04-02"

class VLM():
    def __init__(self):
        self.model_id = "vikhyatk/moondream2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=LATEST_REVISION)
        self.moondream = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, revision=LATEST_REVISION
        ).to(device='mps', dtype=torch.float16)

    def live_cam(self):
        cap = cv2.VideoCapture(0)
        img = None
        i = 0
        while i <= 5:
            ret, frame = cap.read()

            # Wait for 30 milliseconds and check if the user pressed the ESC key
            if i == 5:
                img = frame
                break
            i += 1   
        # Release the video capture device and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img = Image.fromarray(img)  # Convert numpy array to PIL Image
        return img

    def detect_device(self):
        """
        Detects the appropriate device to run on, and return the device and dtype.
        """
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float16
        elif torch.backends.mps.is_available():
            return torch.device("mps"), torch.float16
        else:
            return torch.device("cpu"), torch.float32

    def answer_img_question(self, img, prompt):
        image_embeds = self.moondream.encode_image(img)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        thread = Thread(
            target=self.moondream.answer_question,
            kwargs={
                "image_embeds": image_embeds,
                "question": prompt,
                "tokenizer": self.tokenizer,
                "streamer": streamer,
            },
        )
        thread.start()

        buffer = []
        for new_text in streamer:
            clean_text = re.sub("<$|END$", "", new_text)
            buffer.append(clean_text)

        
        return ''.join(buffer)
        # Join all the yielded values into a single string
        
        
    def webcam_response(self, prompt):
        print("started")
        device, dtype = self.detect_device()
        print("Using device:", device)
        current_time = time()
        img = self.live_cam()
        image_time = time() - current_time
        current_time = time()
        print(f'image generated, time: {image_time:.2f}')
        if img is None:
            print('img empty')      
        print('generating response')
        text = self.answer_img_question(img, prompt)
        vlm_time = time() - current_time
        print(f'text generation took {vlm_time:.2f}')
        return text


if __name__ == "__main__":
    vlmd = VLM()
    text = vlmd.webcam_response("What is in the image and keep it to one sentence")
    print(text)
    pass
