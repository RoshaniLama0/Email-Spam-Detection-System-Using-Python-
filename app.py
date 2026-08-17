import tkinter as tk
import joblib

# Load saved TF-IDF and model
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("spam_model.pkl")


# Check the message
def check_message():
    message = text_box.get("1.0", tk.END).strip()

    if message == "":
        result_label.config(text="")
        confidence_label.config(text="")
        return

    # Convert message to TF-IDF
    message_tfidf = tfidf.transform([message])

    # Make prediction
    prediction = model.predict(message_tfidf)

    # Get confidence
    probabilities = model.predict_proba(message_tfidf)
    confidence = max(probabilities[0]) * 100

    if prediction[0] == 1:
        result_label.config(text="SPAM")
    else:
        result_label.config(text="NOT SPAM")

    confidence_label.config(text=f"{confidence:.2f}%")


# Clear everything
def clear_message():
    text_box.delete("1.0", tk.END)
    result_label.config(text="")
    confidence_label.config(text="")


# -------------------------
# Main Window
# -------------------------

window = tk.Tk()

window.title("Email Spam Detector")
window.geometry("700x650")
window.resizable(False, False)


# Title
title_label = tk.Label(
    window,
    text="EMAIL SPAM DETECTOR",
    font=("Arial", 26, "bold")
)
title_label.pack(pady=25)


# Instruction
instruction_label = tk.Label(
    window,
    text="Enter or paste your message below:",
    font=("Arial", 16)
)
instruction_label.pack(pady=5)


# Message input box
text_box = tk.Text(
    window,
    height=9,
    width=55,
    font=("Arial", 13)
)
text_box.pack(pady=8)


# Check Message button
check_button = tk.Button(
    window,
    text="CHECK MESSAGE",
    font=("Arial", 14, "bold"),
    command=check_message
)
check_button.pack(pady=15)


# Result title
result_title = tk.Label(
    window,
    text="Result:",
    font=("Arial", 20, "bold")
)
result_title.pack(pady=5)


# Result
result_label = tk.Label(
    window,
    text="",
    font=("Arial", 20, "bold")
)
result_label.pack()


# Confidence title
confidence_title = tk.Label(
    window,
    text="Confidence:",
    font=("Arial", 16)
)
confidence_title.pack(pady=10)


# Confidence value
confidence_label = tk.Label(
    window,
    text="",
    font=("Arial", 16)
)
confidence_label.pack()


# Clear button
clear_button = tk.Button(
    window,
    text="CLEAR",
    font=("Arial", 12),
    command=clear_message
)
clear_button.pack(pady=15)


# Start the application
window.mainloop()