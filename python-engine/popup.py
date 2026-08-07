import tkinter as tk
import winsound
import sys
from pathlib import Path
from PIL import Image, ImageTk

def show_popup():
    root = tk.Tk()
    root.overrideredirect(True) # Remove window borders
    root.wm_attributes("-topmost", True) # Always on top
    
    bg_color = "#000000"
    root.configure(bg=bg_color)
    
    # Size and position
    w, h = 400, 120
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w) // 2
    y = sh - h - 50 # 50 px from bottom
    root.geometry(f"{w}x{h}+{x}+{y}")
    
    frame = tk.Frame(root, bg=bg_color)
    frame.pack(expand=True, fill="both")
    
    # Load warning image
    img_path = Path(__file__).parent / "warning.png"
    if img_path.exists():
        try:
            pil_img = Image.open(img_path)
            # Resize to fit nicely at the top
            pil_img.thumbnail((250, 60), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(pil_img)
            img_lbl = tk.Label(frame, image=img, bg=bg_color, borderwidth=0)
            img_lbl.image = img # Keep a reference!
            img_lbl.pack(pady=(10, 5))
        except Exception as e:
            print("Image load error:", e)
        
    text_color = "#ffffff"
    display_text = "SMOKING AND ALCOHOL CONSUMPTION IS\nINJURIOUS TO HEALTH"
    
    lbl = tk.Label(
        frame, 
        text=display_text, 
        fg=text_color, 
        bg=bg_color, 
        font=("Times New Roman", 13, "bold"),
        justify="center"
    )
    lbl.pack(pady=(0, 10))
    
    # Continuously play sound
    def play_sound():
        winsound.Beep(1000, 300)
        root.after(1500, play_sound)
        
    play_sound()
    
    # Clicking it closes it
    def close_app(e):
        root.destroy()
        
    root.bind("<Button-1>", close_app)
    frame.bind("<Button-1>", close_app)
    lbl.bind("<Button-1>", close_app)
    if 'img_lbl' in locals():
        img_lbl.bind("<Button-1>", close_app)
    
    root.mainloop()

if __name__ == "__main__":
    show_popup()
