import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf #type:ignore
from pathlib import Path
import ttkbootstrap as ttk #type:ignore
from ttkbootstrap.constants import * #type:ignore

script_dir = Path(__file__).resolve().parent

default_model_path = script_dir.parent / "Ready_Models" / "model_vgg16.keras"

def main():
    root = ttk.Window(themename="darkly")  
    LandClassifierApp(root)
    root.mainloop()

class LandClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Land Type Classifier - VGG16")
        
        self.img_size = (96, 96)
        self.class_names = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
            'River', 'SeaLake'
        ]
        
        self.show_confidence = True 
        self.model = None
        self.current_image = None
        self.current_image_path = None
        
        self.setup_ui()
        
        if default_model_path.exists():
            try:
                self.model = tf.keras.models.load_model(default_model_path, compile=False) #type:ignore
                self.model_status_label.config( #type:ignore
                    text=f"{default_model_path.name}",
                    bootstyle="success"
                )
            except Exception as e:
                self.model_status_label.config( #type:ignore
                    text="Failed to load default model",
                    bootstyle="danger"
                )
                messagebox.showerror("Error", f"Failed to load default model:\n{e}")

        self.root.update_idletasks()
        self.root.geometry(f"700x{self.root.winfo_reqheight()}")
        
        
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Keras files", "*.keras"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.model_status_label.config( #type:ignore
                    text="Loading...",
                    bootstyle="warning"
                )
                self.root.update()
                
                self.model = tf.keras.models.load_model( #type:ignore
                    file_path,
                    compile=False
                )
                
                model_name = Path(file_path).name
                self.model_status_label.config( #type:ignore
                    text=f"[/] {model_name}",
                    bootstyle="success"
                )
                messagebox.showinfo("Success", "Model loaded successfully!")
                
                if self.current_image_path:
                    self.predict_btn.config(state=NORMAL)
                    
            except Exception as e:
                self.model_status_label.config( #type:ignore
                    text="Failed to load",
                    bootstyle="danger"
                )
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
     
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                img = Image.open(file_path)
                img_resized = img.resize((self.canvas_size, self.canvas_size), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img_resized)
                
                self.image_canvas.delete("all")
            
                self.image_canvas.create_image(0, 0, anchor=NW, image=photo)
                self.image_canvas.image = photo  # Keep a reference #type:ignore
                
                if self.model is not None:
                    self.predict_btn.config(state=NORMAL)
                    
                self.prediction_label.config(text="---")
                if self.confidence_label:
                    self.confidence_label.config(text="---")
                
                if self.model is not None:
                    self.root.after(100, self.predict)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
       
    def setup_ui(self):
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill=BOTH, expand=YES)
        
        title_label = ttk.Label(
            main_container, 
            text="Land Type Classification", 
            font=("Helvetica", 24, "bold"),
            bootstyle="inverse-primary" #type:ignore
        )
        title_label.pack(pady=(0, 20))
        
        model_frame = ttk.Labelframe(
            main_container, 
            text="Model Configuration",
            padding=15,
            bootstyle="info" #type:ignore
        )
        model_frame.pack(fill=X, pady=(0, 20))
        
        status_frame = ttk.Frame(model_frame)
        status_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(
            status_frame, 
            text="Status:", 
            font=("Helvetica", 10, "bold")
        ).pack(side=LEFT)
        
        self.model_status_label = ttk.Label(
            status_frame, 
            text="No model loaded",
            font=("Helvetica", 10),
            bootstyle="danger" #type:ignore
        )
        self.model_status_label.pack(side=LEFT, padx=10)
        
        load_model_btn = ttk.Button(
            model_frame,
            text="Load Model (.keras)",
            command=self.load_model,
            bootstyle="primary", #type:ignore
            width=25
        )
        load_model_btn.pack()
        
        image_frame = ttk.Labelframe(
            main_container,
            text="Image Preview",
            padding=15,
            bootstyle="secondary" #type:ignore
        )
        image_frame.pack(fill=BOTH, expand=YES, pady=(0, 20))
        
        canvas_size = 450
        self.image_canvas = tk.Canvas(
            image_frame,
            width=canvas_size,
            height=canvas_size,
            bg="#2b3e50",
            highlightthickness=0
        )
        self.image_canvas.pack(expand=YES)
        
        self.canvas_size = canvas_size
        
        self.placeholder_text = self.image_canvas.create_text(
            canvas_size // 2, canvas_size // 2,
            text="No image loaded\n\n[!] Click 'Upload Image' to start",
            fill="#95a5a6",
            font=("Helvetica", 14),
            justify="center"
        )
        
        upload_btn = ttk.Button(
            main_container,
            text="Upload Image",
            command=self.upload_image,
            bootstyle="success-outline", #type:ignore
            width=30
        )
        upload_btn.pack(pady=(0, 20))
        
        prediction_frame = ttk.Labelframe(
            main_container,
            text="Prediction Results",
            padding=20,
            bootstyle="warning" #type:ignore
        )
        prediction_frame.pack(fill=X, pady=(0, 20))
        

        pred_label_frame = ttk.Frame(prediction_frame)
        pred_label_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(
            pred_label_frame,
            text="Predicted Class:",
            font=("Helvetica", 11)
        ).pack(side=LEFT)
        
        self.prediction_label = ttk.Label(
            pred_label_frame,
            text="---",
            font=("Helvetica", 16, "bold"),
            bootstyle="warning" #type:ignore
        )
        self.prediction_label.pack(side=LEFT, padx=10)
        
        if self.show_confidence:
            conf_label_frame = ttk.Frame(prediction_frame)
            conf_label_frame.pack(fill=X)
            
            ttk.Label(
                conf_label_frame,
                text="Confidence:",
                font=("Helvetica", 11)
            ).pack(side=LEFT)
            
            self.confidence_label = ttk.Label(
                conf_label_frame,
                text="---",
                font=("Helvetica", 14),
                bootstyle="secondary" #type:ignore
            )
            self.confidence_label.pack(side=LEFT, padx=10)
        else:
            self.confidence_label = None
        
        self.progress = ttk.Progressbar(
            prediction_frame,
            mode='indeterminate',
            bootstyle="success-striped" #type:ignore
        )
        
        self.predict_btn = ttk.Button(
            main_container,
            text="Get Prediction",
            command=self.predict,
            bootstyle="danger", #type:ignore
            width=30,
            state=DISABLED
        )
        self.predict_btn.pack()
        
    
    def predict(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        if self.current_image_path is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        try:
            self.progress.pack(fill=X, pady=10)
            self.progress.start(10)
            self.predict_btn.config(state=DISABLED)
            self.root.update()
            
            img = cv2.imread(self.current_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, self.img_size)
            img_array = np.expand_dims(img_resized, axis=0) / 255.0  # normalize to [0, 1]
            
            pred = self.model.predict(img_array, verbose=0)
            pred_class_idx = np.argmax(pred)
            pred_class = self.class_names[pred_class_idx]
            confidence = pred[0][pred_class_idx] * 100
            
            self.progress.stop()
            self.progress.pack_forget()
            
            self.prediction_label.config(text=pred_class)
            
            if self.confidence_label:
                self.confidence_label.config(text=f"{confidence:.2f}%")
                
                if confidence >= 80:
                    self.confidence_label.config(bootstyle="success") #type:ignore
                elif confidence >= 60:
                    self.confidence_label.config(bootstyle="warning") #type:ignore
                else:
                    self.confidence_label.config(bootstyle="danger") #type:ignore
            
            self.predict_btn.config(state=NORMAL)
            
        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            self.predict_btn.config(state=NORMAL)
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")


if __name__ == "__main__":
    main()