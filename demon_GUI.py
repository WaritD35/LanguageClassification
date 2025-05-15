import tkinter as tk
from tkinter import filedialog
import os
import librosa
from ultralytics import YOLO
import webrtcvad
import matplotlib.pyplot as plt
import numpy as np

def choose_file():
    global file_path, file_name
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        file_label.config(text=os.path.basename(file_path), fg="black")
        file_name = os.path.basename(file_path)
    else:
        file_label.config(text="No file selected")
    return file_path, file_name

def run_selected_model(model_number):
    global file_path, file_name
    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)  # Clear previous output
    output_text.config(state=tk.DISABLED)

    if not file_path:
        output_text.config(state=tk.NORMAL)
        output_text.insert(tk.END, "Please select a .wav file first.\n")
        output_text.config(state=tk.DISABLED)
        return

    # Preprocess the audio file
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # 0-3 (0 = less aggressive, 3 = most aggressive (cut the background))
    
    # Use the global file_name variable
    filename = file_name[:-4]  # Remove the .wav extension
    y, sr = librosa.load(file_path, sr=16000)  # Load the audio file
    y_int16 = (y * 32768).astype(np.int16).tobytes()

    # Divide audio into frames each of 30ms
    frame_duration = 30  # milliseconds
    frame_size = int(sr * frame_duration / 1000) * 2  # *2 because it is 16-bit (2 bytes)
    frames = [y_int16[i:i+frame_size] for i in range(0, len(y_int16), frame_size)]

    # Ensure frames are valid for webrtcvad
    valid_frames = []
    for i in range(0, len(y_int16), frame_size):
        frame = y_int16[i:i + frame_size]
        if len(frame) == frame_size:  # Skip incomplete frames
            valid_frames.append(frame)

    # Check which frames contain speech
    voiced_frames = [frame for frame in valid_frames if vad.is_speech(frame, sr)]
            
    # Combine voiced frames into a single byte string
    speech_pcm = b''.join(voiced_frames)
    speech_np = np.frombuffer(speech_pcm, dtype=np.int16).astype(np.float32) / 32768

    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=speech_np, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibel scale

    # Plot the mel-spectrogram
    plt.figure(figsize=(10, 6))
    plt.axis('off')  # Turn off axis
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')

    # Save the mel-spectrogram plot as an image
    output_dir = "./ProcessedRecording"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    plt.savefig(f"{output_dir}/mel_spec_image_{filename}.png", transparent=True)
    output_image_path = f"{output_dir}/mel_spec_image_{filename}.png"
    plt.close()  # Close the plot to prevent display

    # Load the YOLOv8 model
    model = YOLO("./runs/detect/train/weights/best.pt")  # Load the YOLOv8 model

    #Predict using the model
    results = model.predict(source=output_image_path)
    # Display the results
    for result in results:
        if len(result.boxes.cls) > 0:
            predicted_class = result.boxes.cls[0].item()
            output_text.config(state=tk.NORMAL)
            output_text.insert(tk.END, f"{filename}.wav is predicted as {model.names[int(predicted_class)]}\n")
            output_text.insert(tk.END, f"Confidence: {result.boxes.conf[0].item()}\n")
            output_text.config(state=tk.DISABLED)

file_path, file_name = None, None

# UI
root = tk.Tk()
root.geometry("600x400")
root.resizable(False, False)
root.title("AI Audio Model Demo")
tk.Label(root, text="Yolov8 Model Demonstration with .wav Input", font=("Helvetica", 16)).pack(pady=10)
tk.Button(root, text="Choose .wav File", command=choose_file).pack(pady=5)
file_label = tk.Label(root, text="No file selected", fg="gray")
file_label.pack()
tk.Button(root, text="Predict", command=lambda: run_selected_model(1)).pack(pady=5)
tk.Label(root, text="Output:").pack(pady=5)
output_text = tk.Text(root, height=5, width=70, state=tk.DISABLED)
output_text.pack()
root.mainloop()