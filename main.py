import sys
from collections import Counter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QTextEdit, QMessageBox, QProgressBar
)
import matplotlib.pyplot as plt


class EncryptionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Encryption Tool")
        self.setGeometry(200, 200, 800, 600)

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # Dark mode toggle button
        self.dark_mode_button = QPushButton("Toggle Dark Mode")
        self.dark_mode_button.clicked.connect(self.toggle_dark_mode)
        self.layout.addWidget(self.dark_mode_button)

        # Algorithm selection
        self.algorithm_label = QLabel("Select Algorithm:")
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "Rail Fence Cipher"
        ])
        self.algorithm_combo.currentTextChanged.connect(self.update_ui)
        self.layout.addWidget(self.algorithm_label)
        self.layout.addWidget(self.algorithm_combo)

        # Mode selection
        self.mode_label = QLabel("Select Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Encrypt", "Decrypt"])
        self.layout.addWidget(self.mode_label)
        self.layout.addWidget(self.mode_combo)

        # Number of rails input (only for Rail Fence Cipher)
        self.rails_label = QLabel("Number of Rails:")
        self.rails_input = QLineEdit()
        self.rails_input.setText("2")  # Default number of rails
        self.layout.addWidget(self.rails_label)
        self.layout.addWidget(self.rails_input)

        # Key input (for algorithms that require a key)
        self.key_label = QLabel("Enter Key:")
        self.key_input = QLineEdit()
        self.layout.addWidget(self.key_label)
        self.layout.addWidget(self.key_input)

        # Generate key button (for algorithms that require a key)
        self.generate_key_button = QPushButton("Generate Key")
        self.generate_key_button.clicked.connect(self.generate_key)
        self.layout.addWidget(self.generate_key_button)

        # Plaintext input
        self.plaintext_label = QLabel("Plaintext:")
        self.plaintext_input = QTextEdit()
        self.layout.addWidget(self.plaintext_label)
        self.layout.addWidget(self.plaintext_input)

        # Ciphertext output
        self.ciphertext_label = QLabel("Ciphertext:")
        self.ciphertext_output = QTextEdit()
        self.ciphertext_output.setReadOnly(True)
        self.layout.addWidget(self.ciphertext_label)
        self.layout.addWidget(self.ciphertext_output)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        # Buttons
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_text)
        self.layout.addWidget(self.process_button)

        self.freq_button = QPushButton("Show Frequency Analysis")
        self.freq_button.clicked.connect(self.show_frequency_analysis)
        self.layout.addWidget(self.freq_button)

        # Set initial theme
        self.dark_mode = False
        self.apply_theme()

        # Initialize UI based on the selected algorithm
        self.update_ui()

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def apply_theme(self):
        if self.dark_mode:
            self.main_widget.setStyleSheet("background-color: #2E3440; color: #D8DEE9;")
            self.dark_mode_button.setStyleSheet("background-color: #4C566A; color: #D8DEE9;")
        else:
            self.main_widget.setStyleSheet("background-color: #FFFFFF; color: #000000;")
            self.dark_mode_button.setStyleSheet("background-color: #E5E9F0; color: #000000;")

    def update_ui(self):
        """Update the UI based on the selected algorithm."""
        algorithm = self.algorithm_combo.currentText()

        if algorithm == "Rail Fence Cipher":
            # Hide key input and generate key button
            self.key_label.hide()
            self.key_input.hide()
            self.generate_key_button.hide()
            # Show rails input
            self.rails_label.show()
            self.rails_input.show()
        else:
            # Show key input and generate key button
            self.key_label.show()
            self.key_input.show()
            self.generate_key_button.show()
            # Hide rails input
            self.rails_label.hide()
            self.rails_input.hide()

    def generate_key(self):
        """Generate a key for algorithms that require one."""
        algorithm = self.algorithm_combo.currentText()
        if algorithm == "na":
            QMessageBox.warning(self, "Error", "Please select an algorithm first!")
            return
        else:
            QMessageBox.warning(self, "Error", "Key generation is not supported for this algorithm.")

    def process_text(self):
        """Encrypt or decrypt the text based on the selected algorithm and mode."""
        algorithm = self.algorithm_combo.currentText()
        mode = self.mode_combo.currentText()
        text = self.plaintext_input.toPlainText()

        if not text:
            QMessageBox.warning(self, "Error", "Please enter some text!")
            return

        try:
            if algorithm == "Rail Fence Cipher":
                rails = int(self.rails_input.text())
                result = self.rail_fence_cipher(text, rails, mode)
            else:
                result = "Unsupported algorithm."

            self.ciphertext_output.setPlainText(result)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def show_frequency_analysis(self):
        """Display a histogram of character frequencies in the plaintext."""
        text = self.plaintext_input.toPlainText()
        if not text:
            QMessageBox.warning(self, "Error", "Please enter some text!")
            return

        freq = Counter(text)
        chars = list(freq.keys())
        counts = list(freq.values())

        plt.bar(chars, counts)
        plt.xlabel("Characters")
        plt.ylabel("Frequency")
        plt.title("Character Frequency Analysis")
        plt.show()

    def rail_fence_cipher(self, text, rails, mode):
        """Rail Fence Cipher"""
        if mode == "Encrypt":
            rail = [['\n' for _ in range(len(text))] for _ in range(rails)]
            dir_down = False
            row, col = 0, 0

            for char in text:
                if row == 0 or row == rails - 1:
                    dir_down = not dir_down
                rail[row][col] = char
                col += 1
                row += 1 if dir_down else -1

            result = []
            for i in range(rails):
                for j in range(len(text)):
                    if rail[i][j] != '\n':
                        result.append(rail[i][j])
            return ''.join(result)
        else:
            rail = [['\n' for _ in range(len(text))] for _ in range(rails)]
            dir_down = None
            row, col = 0, 0

            for char in text:
                if row == 0:
                    dir_down = True
                if row == rails - 1:
                    dir_down = False
                rail[row][col] = '*'
                col += 1
                row += 1 if dir_down else -1

            index = 0
            for i in range(rails):
                for j in range(len(text)):
                    if rail[i][j] == '*' and index < len(text):
                        rail[i][j] = text[index]
                        index += 1

            result = []
            row, col = 0, 0
            for _ in range(len(text)):
                if row == 0:
                    dir_down = True
                if row == rails - 1:
                    dir_down = False
                if rail[row][col] != '\n':
                    result.append(rail[row][col])
                    col += 1
                row += 1 if dir_down else -1
            return ''.join(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EncryptionApp()
    window.show()
    sys.exit(app.exec())