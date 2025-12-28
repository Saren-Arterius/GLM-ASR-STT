import os
import sys
import time
import numpy as np
import torch
import torchaudio
import io
import wave
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoModelForSeq2SeqLM, AutoProcessor
import argparse

# --- Configuration ---
MODEL_ID = "zai-org/GLM-ASR-Nano-2512"
TARGET_SAMPLE_RATE = 16000

class ASRServer:
    def __init__(self, port):
        self.port = port
        print("Loading ASR model...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
        self.device_model = self.model.device

    def transcribe(self, audio_data, sample_rate, system_prompt=None, history=None):
        start_time = time.time()
        audio_tensor = torch.from_numpy(audio_data).to(torch.float32)
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
            audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
        
        if system_prompt or history:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)
            
            # The last message should be the user message with the audio
            # Convert torch tensor to numpy array as expected by load_audio in apply_chat_template
            messages.append({"role": "user", "content": [{"type": "audio", "audio": audio_tensor.cpu().numpy()}]})
            
            # Ensure system prompt content is also in list format if the processor expects it
            for msg in messages:
                if isinstance(msg["content"], str):
                    msg["content"] = [{"type": "text", "text": msg["content"]}]
            
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, sampling_rate=TARGET_SAMPLE_RATE)
        else:
            inputs = self.processor.apply_transcription_request(audio_tensor, sampling_rate=TARGET_SAMPLE_RATE)
            
        inputs = {k: v.to(self.device_model) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if hasattr(self.model, "dtype"):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    inputs[k] = v.to(self.model.dtype)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=500)
        
        decoded = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        text = decoded[0] if decoded else ""
        
        duration = time.time() - start_time
        print(f"ASR took {duration:.2f}s")
        return text

    def run(self):
        server_instance = self
        class ASRRequestHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                try:
                    # Try to load as WAV
                    with io.BytesIO(post_data) as bio:
                        with wave.open(bio, 'rb') as wav_file:
                            params = wav_file.getparams()
                            frames = wav_file.readframes(params.nframes)
                            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                            sample_rate = params.framerate
                            
                            if params.nchannels > 1:
                                audio_np = audio_np.reshape(-1, params.nchannels).mean(axis=1)
                except Exception as e:
                    print(f"Error parsing WAV: {e}")
                    self.send_response(400)
                    self.end_headers()
                    return

                # Check for system prompt and history in headers or body
                # For simplicity, we'll check for custom headers first
                system_prompt = self.headers.get('X-System-Prompt')
                # History could be passed as a JSON header, but it's limited in size.
                # For now, let's just support system prompt via header.
                
                text = server_instance.transcribe(audio_np, sample_rate, system_prompt=system_prompt)
                
                self.send_response(200)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(text.encode('utf-8'))

        httpd = HTTPServer(('0.0.0.0', self.port), ASRRequestHandler)
        print(f"HTTP ASR Server listening on port {self.port}...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopping...")
            httpd.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLM-ASR Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    server = ASRServer(args.port)
    server.run()
