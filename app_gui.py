import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, Toplevel
import numpy as np
import pandas as pd # Added for global_x_train etc. type hinting, though not strictly necessary for GUI ops
import logic # Your logic module
import threading # To run the model training without freezing the GUI

# Global variables for data and network (could be used for further actions later)
global_x_train: pd.DataFrame = None
global_y_train: pd.Series = None
global_network: dict = None # Assuming logic returns a dict representing the network

# Global variables for the log window
log_window: Toplevel = None
log_text_widget: scrolledtext.ScrolledText = None

# ---------------- Logger ----------------
def gui_logger(message):
    """Logs messages to the separate log window."""
    global log_text_widget
    if log_text_widget: # Check if the log window and text widget exist
        try:
            log_text_widget.config(state='normal')
            log_text_widget.insert(tk.END, message + "\n")
            log_text_widget.see(tk.END)
            log_text_widget.config(state='disabled')
            # Use after_idle to update the log window from the thread safely
            root.after_idle(root.update_idletasks)
        except tk.TclError:
            # Window might have been closed unexpectedly
            print(f"Log window error: {message}") # Fallback print
    else:
        # If logger is called before window is created or after it's closed
        print(f"Log (no window): {message}") # Fallback print


# ---------------- Log Window Management ----------------
def create_log_window():
    """Creates the separate window for displaying logs."""
    global log_window, log_text_widget
    if log_window is None or not log_window.winfo_exists():
        log_window = Toplevel(root)
        log_window.title("ANN Training Logs")
        # Set geometry relative to the main window if desired, or just default
        # log_window.geometry("600x400")

        log_text_widget = scrolledtext.ScrolledText(log_window, height=15, width=80, state='disabled', wrap='word')
        log_text_widget.pack(padx=10, pady=10, fill="both", expand=True)

        # Prevent closing with X button (optional, but good for controlling lifecycle)
        log_window.protocol("WM_DELETE_WINDOW", lambda: None) # Ignore close attempts
        # Make it modal or transient if desired (e.g., log_window.transient(root))
        return True
    return False # Window already exists

def destroy_log_window():
    """Destroys the separate log window."""
    global log_window, log_text_widget
    if log_window and log_window.winfo_exists():
        log_window.destroy()
    log_window = None
    log_text_widget = None

# ---------------- Browse ----------------
def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

# ---------------- Run Logic in Thread ----------------
def run_models_thread():
    """Function to be run in a separate thread to keep GUI responsive."""
    global global_x_train, global_y_train, global_network

    file_path = file_entry.get()
    try:
        data_percentage = float(data_percentage_entry.get())
        train_percentage = float(train_percentage_entry.get())
        k_value = int(k_entry.get())

        if not (0 < data_percentage <= 100):
            raise ValueError("Data percentage must be between 1 and 100.")
        if not (0 < train_percentage < 100):
            raise ValueError("Training percentage must be between 1 and 99.")
        if not (k_value > 0):
            raise ValueError("K for KNN must be a positive integer.")

    except ValueError as ve:
        messagebox.showerror("Input Error", f"Please enter valid inputs: {ve}")
        destroy_log_window() # Close log window on input error
        results_label.config(text="Input validation failed.")
        return

    if not file_path:
        messagebox.showerror("Input Error", "Please select a CSV file.")
        destroy_log_window() # Close log window if no file selected
        results_label.config(text="No file selected.")
        return

    # Clear previous results/predictions in the main window GUI thread
    root.after_idle(lambda: results_label.config(text="Running models, please wait..."))
    root.after_idle(lambda: prediction_text.config(state='normal'))
    root.after_idle(lambda: prediction_text.delete(1.0, tk.END))
    root.after_idle(lambda: prediction_text.config(state='disabled'))
    root.after_idle(root.update_idletasks)


    results = None
    x_train, y_train, x_test, y_test, network_obj = None, None, None, None, None

    try:
        # Assuming logic.run_models is updated to NOT require min_accuracy_threshold and max_retries
        # and that it correctly uses the provided logger callback.
        # Also assuming logic.run_models returns (results, x_train, y_train, x_test, y_test, network_obj)
        # If your logic.run_models signature is different, adjust the call below.
        results, x_train, y_train, x_test, y_test, network_obj = logic.run_models(
            file_path=file_path,
            data_subset_percentage=data_percentage,
            train_split_percentage=train_percentage,
            k=k_value,
            n_hidden=10, # Default or make it a GUI input if needed
            epochs=1000, # Default or make it a GUI input if needed
            learning_rate=0.1, # Default or make it a GUI input if needed
            logger=gui_logger # Pass the logger function
            # Removed min_accuracy_threshold and max_retries
        )

        # Store global data in the GUI thread after run_models completes
        global_x_train = x_train
        global_y_train = y_train
        global_network = network_obj

    except Exception as e:
        gui_logger(f"An error occurred during model execution: {e}")
        root.after_idle(lambda: messagebox.showerror("Execution Error", f"An error occurred: {e}"))
        root.after_idle(lambda: results_label.config(text="Error during model execution."))
        destroy_log_window()
        return
    finally:
        # Ensure the log window is destroyed after the model runs finish
        destroy_log_window()


    # Update main GUI elements only after the thread completes successfully
    if results is None or results['knn']['metrics'][0] == -1.0 : # Check for error sentinel
        gui_logger("Failed to get valid results from model runs. Check logs.")
        root.after_idle(lambda: messagebox.showerror("Execution Error", "Failed to get valid results from model runs. Check logs."))
        root.after_idle(lambda: results_label.config(text="Failed to get valid results."))
        return

    knn_metrics = results["knn"]["metrics"]
    ann_metrics = results["ann"]["metrics"]
    better_model = results["better_model"]

    results_summary_text = (
        f"Using {data_percentage:.0f}% of CSV data.\n"
        f"Training on {train_percentage:.0f}%, Testing on {100 - train_percentage:.0f}% of the used data.\n\n"
        f"KNN Model (K={k_value}):\n  Accuracy: {knn_metrics[0]:.2f}%\n  Precision: {knn_metrics[1]:.2f}%\n  Recall: {knn_metrics[2]:.2f}%\n\n"
        f"ANN Model:\n  Accuracy: {ann_metrics[0]:.2f}%\n  Precision: {ann_metrics[1]:.2f}%\n  Recall: {ann_metrics[2]:.2f}%\n\n"
        f"Overall Better Model (based on Accuracy): {better_model}"
    )
    root.after_idle(lambda: results_label.config(text=results_summary_text))
    predictions_output_text = (
                "Original Class Labels (for Test Set):\n" + ", ".join(map(str, results["original_labels"])) + "\n\n" + # Display original first
                "KNN Predicted Class Labels (for Test Set):\n" + ", ".join(map(str, results["knn"]["predictions"])) + "\n\n" +
                "ANN Predicted Class Labels (for Test Set):\n" + ", ".join(map(str, results["ann"]["predictions"]))
            )
    root.after_idle(lambda: prediction_text.config(state='normal'))
    root.after_idle(lambda: prediction_text.delete(1.0, tk.END))
    root.after_idle(lambda: prediction_text.insert(tk.END, predictions_output_text))
    root.after_idle(lambda: prediction_text.config(state='disabled'))


# ---------------- Start Thread and Log Window ----------------
def start_run():
    """Starts the log window and then runs the model logic in a separate thread."""
    if create_log_window(): # Only proceed if a new window was created
        # Start the model running function in a separate thread
        thread = threading.Thread(target=run_models_thread)
        thread.start()
    else:
        messagebox.showwarning("Busy", "Model run is already in progress. Please wait.")


# ---------------- GUI Layout ----------------
root = tk.Tk()
root.title("ML Model Comparison GUI (KNN & ANN)")
try:
    root.state('zoomed') # Maximize window if supported
except tk.TclError:
    root.geometry("1000x700") # Fallback size

# --- Configuration Frame ---
config_frame = tk.LabelFrame(root, text="Configuration", padx=10, pady=10)
config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

tk.Label(config_frame, text="CSV File Path:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
file_entry = tk.Entry(config_frame, width=60)
file_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
browse_button = tk.Button(config_frame, text="Browse...", command=browse_file)
browse_button.grid(row=0, column=2, padx=5, pady=5)

tk.Label(config_frame, text="Use % of CSV Data (1-100):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
data_percentage_entry = tk.Entry(config_frame, width=10)
data_percentage_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
data_percentage_entry.insert(0, "100")

tk.Label(config_frame, text="Training Split % (1-99):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
train_percentage_entry = tk.Entry(config_frame, width=10)
train_percentage_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
train_percentage_entry.insert(0, "70") # Default 70%

tk.Label(config_frame, text="K for KNN:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
k_entry = tk.Entry(config_frame, width=10)
k_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
k_entry.insert(0, "3") # Default K

# Removed Min Accuracy Threshold and Max Retries inputs

# Change command to start_run which manages the thread and log window
run_button = tk.Button(config_frame, text="Run Models", command=start_run, bg="lightblue", font=("Arial", 10, "bold"))
run_button.grid(row=4, column=0, columnspan=3, padx=5, pady=10) # Adjusted row

config_frame.columnconfigure(1, weight=1) # Make entry field expand

# --- Results Frame ---
results_frame = tk.LabelFrame(root, text="Results Summary", padx=10, pady=10)
# Adjusted row to 1 (since logs_frame is removed)
results_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

results_label = tk.Label(results_frame, text="Results will appear here.", justify="left", font=("Arial", 10), anchor="nw")
results_label.pack(fill="both", expand=True)

# --- Predictions Frame ---
predictions_frame = tk.LabelFrame(root, text="Predicted Class Labels (Test Set)", padx=10, pady=10)
# Adjusted row to 2 (since logs_frame is removed)
predictions_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

prediction_text = scrolledtext.ScrolledText(predictions_frame, height=10, width=100, wrap='word', state='disabled')
prediction_text.pack(fill="both", expand=True)

# Allow main window rows to expand appropriately
# Row 1 for results, Row 2 for predictions (logs frame removed)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.columnconfigure(0, weight=1) # Allow main column to expand

root.mainloop()