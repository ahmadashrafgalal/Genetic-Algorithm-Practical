import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
from pathlib import Path

from GUI.gui_app import AlgorithmGUI


def launch_gui():
    """Launch the GUI application"""
    try:
        root = tk.Tk()
        root.configure(bg="#f0f0f0")
        
        app = AlgorithmGUI(root)
        
        try:
            root.iconbitmap('icon.ico')
        except:
            pass  
        
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch GUI: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("Genetic Algorithm Visualization Suite")
    print("Advanced Interactive GUI for Algorithm Analysis")
    print("=" * 60)
    print()
    print("Launching GUI... Please wait...")
    print()
    
    launch_gui()
