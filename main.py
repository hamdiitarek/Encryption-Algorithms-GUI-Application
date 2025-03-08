import sys
import math
import random
import requests
from collections import Counter
from PySide6.QtGui import QIcon # type: ignore
from PySide6.QtWidgets import ( # type: ignore
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

        # About Us button
        self.about_button = QPushButton("About Us")
        self.about_button.clicked.connect(self.show_about_us)
        self.layout.addWidget(self.about_button)

        # Dark mode toggle button
        self.dark_mode_button = QPushButton("Toggle Dark Mode")
        self.dark_mode_button.clicked.connect(self.toggle_dark_mode)
        self.layout.addWidget(self.dark_mode_button)

        # Algorithm selection
        self.algorithm_label = QLabel("Select Algorithm:")
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "Rail Fence Cipher",
            "Route Cipher",
            "Playfair Cipher"
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

        # Number of columns input (only for Route Cipher)
        self.columns_label = QLabel("Number of Columns:")
        self.columns_input = QLineEdit()
        self.columns_input.setText("4")  # Default number of columns
        self.layout.addWidget(self.columns_label)
        self.layout.addWidget(self.columns_input)

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

    def show_about_us(self):
            """Show the About Us information."""
            about_text = (
                "Encryption Tool\n"
                "Developed by Hamdi, Omar, Ahmed and Abdelrhman\n\n"
                "This application provides various encryption algorithms including Rail Fence Cipher, Route Cipher, and Playfair Cipher.\n\n"
                "You can encrypt or decrypt text using the selected algorithm and mode.\n\n"
                "You can also generate a random key for the Playfair Cipher algorithm.\n\n"
                "The application also includes a character frequency analysis feature.\n\n"
            )
            QMessageBox.information(self, "About Us", about_text)

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QWidget {
                    background-color: #2E3440;
                    color: #ECEFF4;
                }
                QLabel {
                    color: #ECEFF4;
                }
                QLineEdit, QTextEdit, QComboBox {
                    background-color: #4C566A;
                    color: #ECEFF4;
                    border: 1px solid #5E81AC;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton {
                    background-color: #5E81AC;
                    color: #ECEFF4;
                    border: none;
                    border-radius: 5px;
                    padding: 10px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #81A1C1;
                }
                QPushButton:pressed {
                    background-color: #4C566A;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget {
                    background-color: #F5F5F5;
                    color: #333333;
                }
                QLabel {
                    color: #333333;
                }
                QLineEdit, QTextEdit, QComboBox {
                    background-color: #FFFFFF;
                    color: #333333;
                    border: 1px solid #CCCCCC;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton {
                    background-color: #007BFF;
                    color: #FFFFFF;
                    border: none;
                    border-radius: 5px;
                    padding: 10px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #0056B3;
                }
                QPushButton:pressed {
                    background-color: #004080;
                }
            """)

    def update_ui(self):
        """Update the UI based on the selected algorithm."""
        algorithm = self.algorithm_combo.currentText()

        if algorithm == "Rail Fence Cipher":
            # Hide key input and generate key button
            self.key_label.hide()
            self.key_input.hide()
            self.generate_key_button.hide()
            self.columns_label.hide()
            self.columns_input.hide()
            # Show rails input
            self.rails_label.show()
            self.rails_input.show()
        elif algorithm == "Route Cipher":
            # Hide key input and generate key button
            self.key_label.hide()
            self.key_input.hide()
            self.generate_key_button.hide()
            self.rails_label.hide()
            self.rails_input.hide()
            # Show columns input
            self.columns_label.show()
            self.columns_input.show()
        elif algorithm == "Playfair Cipher":
            # Hide key input and generate key button
            self.rails_label.hide()
            self.rails_input.hide()
            self.columns_label.hide()
            self.columns_input.hide()
            # Show key input
            self.generate_key_button.show()
            self.key_label.show()
            self.key_input.show()
        else:
            # Show key input and generate key button
            self.key_label.show()
            self.key_input.show()
            self.generate_key_button.show()
            # Hide rails input
            self.rails_label.hide()
            self.rails_input.hide()
            self.columns_label.hide()
            self.columns_input.hide()

    def generate_key(self):
        """Generate a key for algorithms that require one."""
        algorithm = self.algorithm_combo.currentText()
        if algorithm == "na":
            QMessageBox.warning(self, "Error", "Please select an algorithm first!")
            return
        elif algorithm == "Playfair Cipher":
            response = requests.get("https://random-word-api.herokuapp.com/word")
            if response.status_code == 200:
                random_key = response.json()[0]
                self.key_input.setText(random_key)
            else:
                QMessageBox.warning(self, "Error", f"API request failed with status code {response.status_code}.")
            
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
            elif algorithm == "Route Cipher":
                columns = int(self.columns_input.text())
                result = self.route_cipher(text, columns, mode)
            elif algorithm == "Playfair Cipher":
                key = self.key_input.text()
                result = self.play_fair_cipher(text, key, mode)
                
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

    def create_playfair_matrix(self, key):
        alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        key = key.upper().replace("J", "I")
        
        seen = set()
        key = "".join([char for char in key if char not in seen and not seen.add(char)])
        
        matrix = []
        
        for char in key:
            if char not in matrix and char in alphabet:
                matrix.append(char)
        
        for char in alphabet:
            if char not in matrix:
                matrix.append(char)
        
        return [matrix[i:i+5] for i in range(0, 25, 5)]
    
    def find_position(self, char, matrix):
        for row in range(5):
            for col in range(5):
                if matrix[row][col] == char:
                    return row, col
        return None
    
    def play_fair_cipher(self, text, key, mode):
        if mode == "Encrypt":
            matrix = self.create_playfair_matrix(key)

            space_positions = [i for i, char in enumerate(text) if char == ' ']
            plaintext = text.upper().replace("J", "I").replace(" ", "")
            digraphs = []
            
            i = 0
            while i < len(plaintext):
                if i + 1 < len(plaintext) and plaintext[i] == plaintext[i + 1]:
                    digraphs.append(plaintext[i] + 'X')
                    i += 1
                else:
                    if i + 1 < len(plaintext):
                        digraphs.append(plaintext[i] + plaintext[i + 1])
                    else:
                        digraphs.append(plaintext[i] + 'X')
                    i += 2
            
            encrypted_text = []
            for digraph in digraphs:
                row1, col1 = self.find_position(digraph[0], matrix)
                row2, col2 = self.find_position(digraph[1], matrix)
                
                if row1 == row2:
                    encrypted_text.append(matrix[row1][(col1 + 1) % 5])
                    encrypted_text.append(matrix[row2][(col2 + 1) % 5])
                
                elif col1 == col2:
                    encrypted_text.append(matrix[(row1 + 1) % 5][col1])
                    encrypted_text.append(matrix[(row2 + 1) % 5][col2])
                
                else:
                    encrypted_text.append(matrix[row1][col2])
                    encrypted_text.append(matrix[row2][col1])
            
            encrypted_with_spaces = list(''.join(encrypted_text))
            for pos in space_positions:
                encrypted_with_spaces.insert(pos, ' ')
            
            return ''.join(encrypted_with_spaces)
        else :
            matrix = self.create_playfair_matrix(key)

            space_positions = [i for i, char in enumerate(text) if char == ' ']
            ciphertext = text.upper().replace(" ", "")
            
            decrypted_text = []
            for i in range(0, len(ciphertext), 2):
                if i + 1 < len(ciphertext):
                    digraph = ciphertext[i:i+2]
                    row1, col1 = self.find_position(digraph[0], matrix)
                    row2, col2 = self.find_position(digraph[1], matrix)
                    
                    if row1 == row2:
                        decrypted_text.append(matrix[row1][(col1 - 1) % 5])
                        decrypted_text.append(matrix[row2][(col2 - 1) % 5])
                    
                    elif col1 == col2:
                        decrypted_text.append(matrix[(row1 - 1) % 5][col1])
                        decrypted_text.append(matrix[(row2 - 1) % 5][col2])
                    
                    else:
                        decrypted_text.append(matrix[row1][col2])
                        decrypted_text.append(matrix[row2][col1])
                        
            if decrypted_text[len(decrypted_text) - 1] == 'X' :
                decrypted_text.pop(len(decrypted_text) - 1)
            
            decrypted_with_spaces = list(''.join(decrypted_text))
            i = 0
            while i < len(decrypted_with_spaces) - 2:
                if decrypted_with_spaces[i] == decrypted_with_spaces[i + 2] and decrypted_with_spaces[i + 1] == 'X':
                    decrypted_with_spaces.pop(i + 1)
                i += 1

            for pos in space_positions:
                if pos < len(decrypted_with_spaces):
                    decrypted_with_spaces.insert(pos, ' ')
            
            return ''.join(decrypted_with_spaces)

    def route_cipher(self, text, columns, mode):
        if mode == "Encrypt":
            space_positions = [index for index, char in enumerate(text) if char == ' ']
            
            plaintext = text.replace(" ", "")

            num_rows = math.ceil(len(plaintext) / columns)
            
            grid = [['' for _ in range(columns)] for _ in range(num_rows)]
            
            index = 0
            for row in range(num_rows):
                for col in range(columns):
                    if index < len(plaintext):
                        grid[row][col] = plaintext[index]
                        index += 1
            
            char = 'v'
            for col in range(columns):
                if grid[num_rows-1][col] == '':
                    grid[num_rows-1][col] = char

            encrypted_text = []
            for col in range(columns):
                for row in range(num_rows):
                    if grid[row][col] != '':
                        encrypted_text.append(grid[row][col])
            
            encrypted_text = ''.join(encrypted_text)
            
            encrypted_text_with_spaces = list(encrypted_text)
            for pos in space_positions:
                encrypted_text_with_spaces.insert(pos, ' ')
            
            return ''.join(encrypted_text_with_spaces)
        else:
            space_positions = [index for index, char in enumerate(text) if char == ' ']

            ciphertext = text.replace(" ", "")


            num_rows = math.ceil(len(ciphertext) / columns)
            
            grid = [['' for _ in range(columns)] for _ in range(num_rows)]
            
            index = 0
            for col in range(columns):
                for row in range(num_rows):
                    if index < len(ciphertext):
                        grid[row][col] = ciphertext[index]
                        index += 1
            
            decrypted_text = []
            for row in range(num_rows):
                for col in range(columns):
                    if grid[row][col] != '':
                        decrypted_text.append(grid[row][col])

            while decrypted_text and decrypted_text[-1] == 'v':
                decrypted_text.pop()

            decrypted_text = ''.join(decrypted_text)

            decrypted_text_with_spaces = list(decrypted_text)
            for pos in space_positions:
                decrypted_text_with_spaces.insert(pos, ' ')
            
            return ''.join(decrypted_text_with_spaces)

    def rail_fence_cipher(self, text, rails, mode):
        """Rail Fence Cipher"""
        if mode == "Encrypt":
            rail_pattern = ['' for _ in range(rails)]
            
            current_rail = 0
            direction = 1  
            
            for char in text:
                rail_pattern[current_rail] += char
                current_rail += direction
                
                if current_rail == 0 or current_rail == rails - 1:
                    direction *= -1
            
            return ''.join(rail_pattern)
        else:
            rail_pattern = [['' for _ in range(len(text))] for _ in range(rails)]
    
            current_rail = 0
            direction = 1  
            
            for i in range(len(text)):
                rail_pattern[current_rail][i] = '*'
                current_rail += direction
                
                if current_rail == 0 or current_rail == rails - 1:
                    direction *= -1
            
            index = 0
            for i in range(rails):
                for j in range(len(text)):
                    if rail_pattern[i][j] == '*' and index < len(text):
                        rail_pattern[i][j] = text[index]
                        index += 1
            
            result = []
            current_rail = 0
            direction = 1
            
            for j in range(len(text)):
                result.append(rail_pattern[current_rail][j])
                current_rail += direction
                
                if current_rail == 0 or current_rail == rails - 1:
                    direction *= -1
            
            return ''.join(result)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EncryptionApp()
    app.setWindowIcon(QIcon("encryptiontoolLogo.icns"))
    app.setApplicationName("Encryption Tool")
    window.setWindowIcon(QIcon("encryptiontoolLogo.icns"))    
    window.setWindowTitle("Encryption Tool")
    window.show()
    sys.exit(app.exec())