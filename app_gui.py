import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

import logic  # Your logic module

global_x_train = None
global_y_train = None
global_network = None

# ---------------- Logger ----------------
def gui_logger(message):
    ann_progress_text.config(state='normal')
    ann_progress_text.insert(tk.END, message + "\n")
    ann_progress_text.see(tk.END)
    ann_progress_text.config(state='disabled')
    root.update_idletasks()

# ---------------- Browse ----------------
def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

# ---------------- Run ----------------
def run_models():
    global global_x_train, global_y_train, global_network

    file_path = file_entry.get()
    try:
        data_percentage = float(data_percentage_entry.get())
        train_percentage = float(train_percentage_entry.get())

        if not (0 < data_percentage <= 100) or not (0 < train_percentage < 100):
            raise ValueError
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid percentages (1–100 for data, 1–99 for training).")
        return

    if not file_path:
        messagebox.showerror("Input Error", "Please select a CSV file.")
        return

    try:
        results, x_train, y_train, x_test, y_test, network_obj = logic.run_models(
            file_path=file_path,
            data_subset_percentage=data_percentage,
            train_split_percentage=train_percentage,
            k=3,
            n_hidden=10,
            epochs=1000,
            learning_rate=0.1,
            logger=gui_logger
        )
    except Exception as e:
        messagebox.showerror("Execution Error", f"An error occurred: {e}")
        return

    global_x_train = x_train
    global_y_train = y_train
    global_network = network_obj

    knn_metrics = results["knn"]["metrics"]
    ann_metrics = results["ann"]["metrics"]
    better_model = results["better_model"]

    results_text = (
        f"Using {data_percentage:.0f}% of CSV\n"
        f"Training on {train_percentage:.0f}%, Testing on {100 - train_percentage:.0f}% of selected data\n\n"
        f"KNN Model:\n  Accuracy: {knn_metrics[0]:.2f}%\n  Precision: {knn_metrics[1]:.2f}%\n  Recall: {knn_metrics[2]:.2f}%\n\n"
        f"ANN Model:\n  Accuracy: {ann_metrics[0]:.2f}%\n  Precision: {ann_metrics[1]:.2f}%\n  Recall: {ann_metrics[2]:.2f}%\n\n"
        f"Better Model: {better_model}"
    )
    results_label.config(text=results_text)

    predictions_text = (
        "KNN Predicted Labels:\n" + ", ".join(map(str, results["knn"]["predictions"])) + "\n\n" +
        "ANN Predicted Labels:\n" + ", ".join(map(str, results["ann"]["predictions"]))
    )
    prediction_text.config(state='normal')
    prediction_text.delete(1.0, tk.END)
    prediction_text.insert(tk.END, predictions_text)
    prediction_text.config(state='disabled')

# ---------------- GUI Layout ----------------
root = tk.Tk()
root.title("ML Model GUI")
root.state('zoomed')  # Maximize

# File selection
tk.Label(root, text="CSV File Path:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
file_entry = tk.Entry(root, width=50)
file_entry.grid(row=0, column=1, padx=5, pady=5)
browse_button = tk.Button(root, text="Browse...", command=browse_file)
browse_button.grid(row=0, column=2, padx=5, pady=5)

# Data percentage
tk.Label(root, text="Use this % of CSV Data (1-100):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
data_percentage_entry = tk.Entry(root, width=10)
data_percentage_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
data_percentage_entry.insert(0, "100")

# Training split
tk.Label(root, text="Training Split % (1-99 of used data):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
train_percentage_entry = tk.Entry(root, width=10)
train_percentage_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
train_percentage_entry.insert(0, "75")

# Run Button
run_button = tk.Button(root, text="Run Models", command=run_models)
run_button.grid(row=3, column=1, padx=5, pady=10)

# Output display
results_label = tk.Label(root, text="", justify="left", font=("Arial", 10))
results_label.grid(row=4, column=0, columnspan=3, padx=5, pady=5)

# ANN log
tk.Label(root, text="ANN Training Progress:").grid(row=5, column=0, columnspan=3, sticky="w", padx=5)
ann_progress_text = tk.Text(root, height=10, width=120, state='disabled', wrap='word')
ann_progress_text.grid(row=6, column=0, columnspan=3, padx=5, pady=5)

# Predictions
tk.Label(root, text="Predicted Class Labels:").grid(row=7, column=0, columnspan=3, sticky="w", padx=5)
prediction_text = tk.Text(root, height=10, width=120, wrap='word')
prediction_text.grid(row=8, column=0, columnspan=3, padx=5, pady=10)
prediction_text.config(state='disabled')

root.mainloop()
