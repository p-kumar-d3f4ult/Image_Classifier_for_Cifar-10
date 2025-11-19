import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk, ImageOps
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('cifar10_cnn.h5')

# CIFAR-10 class names
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

IMG_SIZE = (32, 32)

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = ImageOps.fit(img, IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

def predict_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image to Classify",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if file_path:
        # Show loading state
        result_label.config(text="🔄 Analyzing image...", fg="#ffd23f")
        root.update()
        
        input_img, display_img = preprocess_image(file_path)
        pred = model.predict(input_img)
        class_idx = np.argmax(pred)
        predicted_class = class_names[class_idx]
        confidence = np.max(pred) * 100

        # Update image panel
        display_img = display_img.resize((180, 180))
        img_tk = ImageTk.PhotoImage(display_img)
        panel.config(image=img_tk, text="")
        panel.image = img_tk

        # Update result with color coding
        result_text = f"🎯 Prediction: {predicted_class.upper()}\n📊 Confidence: {confidence:.2f}%"
        
        # Color coding based on confidence
        if confidence >= 80:
            result_color = "#27ae60"  # Green for high confidence
        elif confidence >= 60:
            result_color = "#f39c12"  # Orange for medium confidence
        else:
            result_color = "#e74c3c"  # Red for low confidence
            
        result_label.config(text=result_text, fg="#ffffff", bg=result_color)

# Create main window
root = tk.Tk()
root.title("🤖 CIFAR-10 AI Image Classifier")
root.geometry("800x700")
root.configure(bg="#0f0f23")
root.resizable(True, True)

# Title Section
title_frame = Frame(root, bg="#0f0f23")
title_frame.pack(pady=30)

main_title = Label(title_frame, text="CIFAR-10", 
                  font=("Arial", 32, "bold"),
                  bg="#0f0f23", fg="#64b5f6")
main_title.pack()

subtitle = Label(title_frame, text="AI Image Classifier", 
                font=("Arial", 16),
                bg="#0f0f23", fg="#90caf9")
subtitle.pack(pady=(5, 0))

description = Label(title_frame, text="Upload an image and let AI identify what it sees", 
                   font=("Arial", 11),
                   bg="#0f0f23", fg="#b8c1ec")
description.pack(pady=(8, 0))

# Image Section
image_frame = Frame(root, bg="#0f0f23")
image_frame.pack(pady=20)

image_title = Label(image_frame, text="📸 Selected Image", 
                   font=("Arial", 12, "bold"),
                   bg="#0f0f23", fg="#64b5f6")
image_title.pack(pady=(0, 10))

# Image container
image_container = Frame(image_frame, bg="#333366", bd=2, relief="solid")
image_container.pack()

panel = Label(image_container, 
             bg="#1a1a3e", 
             text="📁\n\nNo image selected\n\nClick 'Select Image' below", 
             font=("Arial", 11), 
             fg="#b8c1ec", 
             width=25, 
             height=12,
             justify="center")
panel.pack(padx=10, pady=10)

# Button Section - This is the most important part!
button_frame = Frame(root, bg="#0f0f23")
button_frame.pack(pady=20)

# Create button with simple styling first
select_btn = Button(button_frame, 
                   text="📷 Select Image", 
                   command=predict_image,
                   font=("Arial", 12, "bold"), 
                   bg="#eebbc3", 
                   fg="#0f0f23", 
                   padx=25, 
                   pady=10,
                   cursor="hand2")
select_btn.pack()

# Results Section
results_frame = Frame(root, bg="#0f0f23")
results_frame.pack(pady=20, fill="x")

result_title = Label(results_frame, text="🎯 Results", 
                    font=("Arial", 12, "bold"),
                    bg="#0f0f23", fg="#64b5f6")
result_title.pack(pady=(0, 10))

result_label = Label(results_frame, 
                    text="🤔 Waiting for an image...", 
                    font=("Arial", 12, "bold"),
                    bg="#333366", 
                    fg="#b8c1ec", 
                    pady=12)
result_label.pack(padx=60, fill="x")

# Classes Section
classes_frame = Frame(root, bg="#0f0f23")
classes_frame.pack(pady=20, fill="x")

classes_title = Label(classes_frame, text="🎯 Detectable Objects", 
                     font=("Arial", 12, "bold"),
                     bg="#0f0f23", fg="#64b5f6")
classes_title.pack(pady=(0, 8))

classes_text = " • ".join([name.title() for name in class_names])
classes_label = Label(classes_frame, text=classes_text, 
                     font=("Arial", 10),
                     bg="#1a1a3e", fg="#90caf9",
                     wraplength=600, justify="center",
                     pady=10)
classes_label.pack(padx=40, fill="x")

# Footer
footer_frame = Frame(root, bg="#0f0f23")
footer_frame.pack(side="bottom", pady=20)

footer = Label(footer_frame, 
              text="Created with ❤️ by Piyush Kumar | Powered by TensorFlow", 
              font=("Arial", 9), 
              bg="#0f0f23", 
              fg="#64b5f6")
footer.pack()

# Test the button immediately
print("Button created successfully!")

root.mainloop()