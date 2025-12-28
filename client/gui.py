import os
import sys
import threading
import time
import tkinter as tk
import customtkinter as ctk
import numpy as np
from PIL import Image
import sounddevice as sd

# Import the existing ASRClient logic
try:
    from main import ASRClient, CONFIG, play_sound, save_config
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from main import ASRClient, CONFIG, play_sound, save_config

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ASRGui(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GLM-ASR Client")
        self.geometry("1000x600")

        self.client = None
        self.local_server_proc = None
        self.is_running = False
        self.volume_stream = None

        self.setup_ui()
        self.update_status("Disconnected")

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="GLM-ASR", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.start_button = ctk.CTkButton(self.sidebar_frame, text="Start Client", command=self.toggle_client)
        self.start_button.grid(row=1, column=0, padx=20, pady=10)

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.appearance_mode_optionemenu.set("Dark")

        # Main Content
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Status
        self.status_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.status_frame.grid(row=0, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self.status_frame, text="Status: Disconnected", font=ctk.CTkFont(size=16))
        self.status_label.grid(row=0, column=0, sticky="w")

        self.clear_button = ctk.CTkButton(self.status_frame, text="Clear Log", width=80, command=self.clear_log)
        self.clear_button.grid(row=0, column=1, sticky="e")

        # Transcription Area
        self.textbox = ctk.CTkTextbox(self.main_frame, width=400, height=200)
        self.textbox.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.textbox.insert("0.0", "Transcriptions will appear here...\n")
        self.textbox.configure(state="disabled")

        # Settings
        self.settings_frame = ctk.CTkFrame(self.main_frame)
        self.settings_frame.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        self.settings_frame.grid_columnconfigure(1, weight=1)

        self.use_custom_url = ctk.BooleanVar(value=not CONFIG.get("use_local_server", True))
        self.url_checkbox = ctk.CTkCheckBox(self.settings_frame, text="ASR Server URL", variable=self.use_custom_url, command=self.toggle_url_entry)
        self.url_checkbox.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.server_entry = ctk.CTkEntry(self.settings_frame)
        self.server_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        if not self.use_custom_url.get():
            self.server_entry.insert(0, "Will start a local server automatically")
            self.server_entry.configure(state="disabled", text_color="grey")
        else:
            self.server_entry.insert(0, CONFIG.get("default_asr_server", "http://localhost:8000"))
            self.server_entry.configure(state="normal")

        ctk.CTkLabel(self.settings_frame, text="Input Device:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        devices = sd.query_devices()
        device_names = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        self.device_option = ctk.CTkOptionMenu(self.settings_frame, values=device_names, command=self.on_device_change)
        self.device_option.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        self.volume_label = ctk.CTkLabel(self.settings_frame, text="Volume:")
        self.volume_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.volume_meter = ctk.CTkProgressBar(self.settings_frame)
        self.volume_meter.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.volume_meter.set(0)

        ctk.CTkLabel(self.settings_frame, text="Hotkey:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        
        self.hotkey_frame = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.hotkey_frame.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        self.hotkey_frame.grid_columnconfigure(0, weight=1)

        common_keys = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "caps lock", "insert", "home", "page up", "page down", "end"]
        current_hotkey = CONFIG.get("hotkey", "f12")
        if current_hotkey not in common_keys:
            common_keys.append(current_hotkey)
        
        self.hotkey_option = ctk.CTkOptionMenu(self.hotkey_frame, values=common_keys)
        self.hotkey_option.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="ew")
        self.hotkey_option.set(current_hotkey)

        self.record_button = ctk.CTkButton(self.hotkey_frame, text="Record", width=60, command=self.start_recording_hotkey)
        self.record_button.grid(row=0, column=1, padx=0, pady=0)

        ctk.CTkLabel(self.settings_frame, text="System Prompt:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.system_prompt_entry = ctk.CTkEntry(self.settings_frame)
        self.system_prompt_entry.grid(row=4, column=1, padx=10, pady=5, sticky="ew")
        self.system_prompt_entry.insert(0, CONFIG.get("system_prompt", ""))

        self.disable_log_var = ctk.BooleanVar(value=CONFIG.get("disable_log", False))
        self.disable_log_checkbox = ctk.CTkCheckBox(self.settings_frame, text="Disable Transcribe Log", variable=self.disable_log_var)
        self.disable_log_checkbox.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        self.save_button = ctk.CTkButton(self.settings_frame, text="Save Config", command=self.save_config_event)
        self.save_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)
        
        # Try to select default from config
        default_device = self.find_default_device_index(CONFIG.get("audio_devices", []))
        if default_device is not None:
            for name in device_names:
                if name.startswith(f"{default_device}:"):
                    self.device_option.set(name)
                    break
        
        self.start_volume_monitor()

    def on_device_change(self, _):
        self.start_volume_monitor()

    def start_volume_monitor(self):
        if self.is_running:
            return
        # Use a lock to prevent multiple threads from starting/stopping the monitor simultaneously
        if not hasattr(self, "volume_monitor_lock"):
            self.volume_monitor_lock = threading.Lock()
            
        threading.Thread(target=self._start_volume_monitor_thread, daemon=True).start()

    def stop_volume_monitor(self):
        if not hasattr(self, "volume_monitor_lock"):
            self.volume_monitor_lock = threading.Lock()
            
        # Don't block the main thread waiting for the lock
        threading.Thread(target=self._stop_volume_monitor_thread, daemon=True).start()

    def _stop_volume_monitor_thread(self):
        with self.volume_monitor_lock:
            if self.volume_stream:
                try:
                    self.volume_stream.stop()
                    self.volume_stream.close()
                except:
                    pass
                self.volume_stream = None

    def _start_volume_monitor_thread(self):
        # Use a timeout or non-blocking acquire to avoid deadlocks
        acquired = self.volume_monitor_lock.acquire(timeout=2.0)
        if not acquired:
            print("Volume monitor lock acquisition timed out")
            return
            
        try:
            if self.volume_stream:
                try:
                    self.volume_stream.stop()
                    self.volume_stream.close()
                except:
                    pass
                self.volume_stream = None

            try:
                device_str = self.device_option.get()
                if not device_str or ":" not in device_str:
                    return
                device_id = int(device_str.split(":")[0])
                
                def audio_callback(indata, frames, time, status):
                    if not self.volume_stream:
                        return
                    volume_norm = np.linalg.norm(indata) * 10
                    # Scale to 0-1 for progress bar
                    level = min(1.0, volume_norm / 100)
                    self.after(0, lambda: self.volume_meter.set(level))

                # Use a small blocksize to avoid blocking
                self.volume_stream = sd.InputStream(device=device_id, channels=1, callback=audio_callback, blocksize=1024)
                self.volume_stream.start()
            except Exception as e:
                print(f"Error starting volume monitor: {e}")
                self.volume_stream = None
        finally:
            self.volume_monitor_lock.release()

    def find_default_device_index(self, name_substrings):
        devices = sd.query_devices()
        for sub in name_substrings:
            for i, dev in enumerate(devices):
                if sub.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                    return i
        return None

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def toggle_url_entry(self):
        if self.use_custom_url.get():
            self.server_entry.configure(state="normal", text_color=ctk.ThemeManager.theme["CTkEntry"]["text_color"])
            # Restore previous value if it was the placeholder
            if self.server_entry.get() == "Will start a local server automatically":
                self.server_entry.delete(0, "end")
                self.server_entry.insert(0, CONFIG.get("default_asr_server", "http://localhost:8000"))
        else:
            self.server_entry.delete(0, "end")
            self.server_entry.insert(0, "Will start a local server automatically")
            self.server_entry.configure(state="disabled", text_color="grey")

    def update_status(self, status, color=None):
        self.status_label.configure(text=f"Status: {status}")
        if color:
            self.status_label.configure(text_color=color)
        else:
            self.status_label.configure(text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"])

    def log_transcription(self, text):
        if self.disable_log_var.get() and "Config saved" not in text:
            return
        self.textbox.configure(state="normal")
        self.textbox.insert("end", f"> {text}\n")
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def clear_log(self):
        self.textbox.configure(state="normal")
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", "Transcriptions will appear here...\n")
        self.textbox.configure(state="disabled")

    def start_recording_hotkey(self):
        self.record_button.configure(text="...", fg_color="orange")
        self.bind("<Key>", self.on_key_pressed)
        self.focus_set()

    def on_key_pressed(self, event):
        # Unbind immediately
        self.unbind("<Key>")
        
        # Map tkinter keysym to keyboard library names if necessary
        key = event.keysym.lower()
        
        # Common mappings
        mapping = {
            "caps_lock": "caps lock",
            "next": "page down",
            "prior": "page up",
            "return": "enter",
            "control_l": "ctrl",
            "control_r": "ctrl",
            "alt_l": "alt",
            "alt_r": "alt",
            "shift_l": "shift",
            "shift_r": "shift",
        }
        key = mapping.get(key, key)
        
        # Update option menu
        current_values = self.hotkey_option.cget("values")
        if key not in current_values:
            self.hotkey_option.configure(values=list(current_values) + [key])
        
        self.hotkey_option.set(key)
        self.record_button.configure(text="Record", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        print(f"Recorded hotkey: {key}")

    def toggle_client(self):
        if not self.is_running:
            self.start_client()
        else:
            self.stop_client()

    def save_config_event(self):
        use_custom = self.use_custom_url.get()
        server_url = self.server_entry.get()
        hotkey = self.hotkey_option.get()
        system_prompt = self.system_prompt_entry.get()
        disable_log = self.disable_log_var.get()

        CONFIG["use_local_server"] = not use_custom
        CONFIG["hotkey"] = hotkey
        if use_custom:
            CONFIG["default_asr_server"] = server_url
        CONFIG["system_prompt"] = system_prompt
        CONFIG["disable_log"] = disable_log
        save_config(CONFIG)
        self.log_transcription("Config saved.")
        print("Config saved.")

    def start_client(self):
        self.stop_volume_monitor()
        
        if not self.use_custom_url.get():
            server_url = CONFIG.get("default_asr_server", "http://localhost:8000")
        else:
            server_url = self.server_entry.get()
            
        device_str = self.device_option.get()
        if not device_str or ":" not in device_str:
            self.update_status("Error: No device selected", "red")
            return
        
        try:
            device_id = int(device_str.split(":")[0])
        except ValueError:
            self.update_status("Error: Invalid device ID", "red")
            return

        hotkey = self.hotkey_option.get()

        # Update config in memory
        CONFIG["hotkey"] = hotkey
        CONFIG["default_asr_server"] = server_url
        CONFIG["system_prompt"] = self.system_prompt_entry.get()

        self.is_running = True
        self.start_button.configure(text="Stop Client", fg_color="red")
        self.update_status("Starting...", "orange")

        self.client_thread = threading.Thread(target=self.run_client, args=(server_url, device_id), daemon=True)
        self.client_thread.start()

    def run_client(self, server_url, device_id):
        try:
            self.after(0, lambda: self.update_status("Loading VAD Model...", "orange"))
            self.client = ASRClient(server_url)
            # Override device from UI
            try:
                dev_info = sd.query_devices(device_id)
                self.client.input_device = device_id
                self.client.input_sample_rate = int(dev_info['default_samplerate'])
                self.client.input_channels = dev_info['max_input_channels']
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self.update_status(f"Error: Invalid device - {msg}", "red"))
                self.after(0, self.stop_client)
                return

            # Monkey patch the client to update UI
            original_send_to_asr = self.client.send_to_asr
            def patched_send_to_asr(audio_data, sample_rate):
                self.after(0, lambda: self.update_status("Processing...", "orange"))
                res = original_send_to_asr(audio_data, sample_rate)
                if res:
                    self.after(0, lambda r=res: self.log_transcription(r))
                self.after(0, lambda: self.update_status("Listening", "green"))
                return res
            
            self.client.send_to_asr = patched_send_to_asr

            # Start local server if needed
            if CONFIG.get("use_local_server", True):
                self.local_server_proc = self.client.start_local_server()
            
            # Start threads
            threading.Thread(target=self.client.socket_listener, daemon=True).start()
            threading.Thread(target=self.client.recording_loop, daemon=True).start()
            
            # Start keyboard listener
            self.client.start_keyboard_subprocess()
            
            # Wait for server to be ready
            self.after(0, lambda: self.update_status("Waiting for ASR Server...", "orange"))
            
            # Check server readiness without blocking the whole thread if possible, 
            # but since this is already in a thread, it's fine to sleep here.
            while not self.client.stop_event.is_set():
                if self.client.check_server_ready():
                    break
                time.sleep(1)
            
            if self.client.stop_event.is_set():
                return

            self.after(0, lambda: self.update_status("Listening", "green"))

            # Start audio input stream
            CHUNK_SIZE = int(self.client.input_sample_rate * 32 / 1000)
            try:
                stream = sd.InputStream(device=self.client.input_device, 
                                    samplerate=self.client.input_sample_rate, 
                                    channels=self.client.input_channels, 
                                    callback=self.client.audio_callback, 
                                    blocksize=CHUNK_SIZE)
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self.update_status(f"Audio Error: {msg}", "red"))
                self.is_running = False
                return

            with stream:
                while not self.client.stop_event.is_set():
                    # Check if recording is active to update UI status
                    if self.client.is_recording_dict.get("internal_active"):
                        self.after(0, lambda: self.update_status("Recording...", "red"))
                    elif self.is_running:
                        # Only set to listening if we weren't just processing
                        current_status = self.status_label.cget("text")
                        if current_status == "Status: Recording...":
                             self.after(0, lambda: self.update_status("Listening", "green"))
                    
                    time.sleep(0.5)
        except Exception as e:
            error_msg = str(e)
            print(f"Client error: {error_msg}")
            self.after(0, lambda msg=error_msg: self.update_status(f"Error: {msg}", "red"))

    def stop_client(self):
        if self.client:
            self.client.stop_event.set()
        self.is_running = False
        self.start_button.configure(text="Start Client", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        self.update_status("Disconnected")
        self.start_volume_monitor()

    def on_closing(self):
        self.stop_client()
        if self.volume_stream:
            self.volume_stream.stop()
            self.volume_stream.close()
        self.destroy()

if __name__ == "__main__":
    app = ASRGui()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
