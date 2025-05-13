import sys
import math
import random
import numpy as np  
import requests
from collections import Counter
from PySide6.QtGui import QIcon # type: ignore
from PySide6.QtWidgets import ( # type: ignore
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QTextEdit, QMessageBox, QProgressBar
)

import matplotlib.pyplot as plt
import base64
import binascii


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
            "Playfair Cipher",
            "Hill Cipher",
            "Vigenere Cipher",
            "Caesar Cipher",
            "One-Time Pad",
            "Simplified DES",
            "DES",
            "AES",
            "Euclidean Algorithm",
            "Extended Euclidean Algorithm"
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
            self.mode_label.show()
            self.mode_combo.show()
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
            self.mode_label.show()
            self.mode_combo.show()
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
            self.mode_label.show()
            self.mode_combo.show()
        elif algorithm == "Hill Cipher":
            # Hide rails and columns input
            self.rails_label.hide()
            self.rails_input.hide()
            self.columns_label.hide()
            self.columns_input.hide()
            # Show key input
            self.generate_key_button.hide()
            self.key_label.show()
            self.key_input.show()
            self.mode_label.show()
            self.mode_combo.show()
        elif algorithm == "Vigenere Cipher":
            # Hide rails and columns input
            self.rails_label.hide()
            self.rails_input.hide()
            self.columns_label.hide()
            self.columns_input.hide()
            # Show key input
            self.generate_key_button.hide()
            self.key_label.show()
            self.key_input.show()
            self.mode_label.show()
            self.mode_combo.show()
        elif algorithm == "Caesar Cipher":
            # Hide rails and columns input
            self.rails_label.hide()
            self.rails_input.hide()
            self.columns_label.hide()
            self.columns_input.hide()
            # Show key input (used for shift value)
            self.generate_key_button.hide()
            self.key_label.setText("Enter Shift:")
            self.key_label.show()
            self.key_input.show()
            self.mode_label.show()
            self.mode_combo.show()
        elif algorithm == "One-Time Pad":
            # Hide rails and columns input
            self.rails_label.hide()
            self.rails_input.hide()
            self.columns_label.hide()
            self.columns_input.hide()
            # Show key input
            self.generate_key_button.show()
            self.key_label.setText("Enter Key:")
            self.key_label.show()
            self.key_input.show()
            self.mode_label.show()
            self.mode_combo.show()
        elif algorithm == "Simplified DES":
            # Hide rails and columns input
            self.rails_label.hide()
            self.rails_input.hide()
            self.columns_label.hide()
            self.columns_input.hide()
            # Show key input
            self.generate_key_button.show()
            self.key_label.setText("Enter 10-bit Key (binary):")
            self.key_label.show()
            self.key_input.show()
            self.mode_label.show()
            self.mode_combo.show()
        elif algorithm == "DES":
            # Hide rails and columns input
            self.rails_label.hide()
            self.rails_input.hide()
            self.columns_label.hide()
            self.columns_input.hide()
            # Show key input
            self.generate_key_button.show()
            self.key_label.setText("Enter 64-bit Key:")
            self.key_label.show()
            self.key_input.show()
            self.mode_label.show()
            self.mode_combo.show()
        elif algorithm == "AES":
            # Hide rails and columns input
            self.rails_label.hide()
            self.rails_input.hide()
            self.columns_label.hide()
            self.columns_input.hide()
            # Show key input
            self.generate_key_button.show()
            self.key_label.setText("Enter 128-bit Key:")
            self.key_label.show()
            self.key_input.show()
            self.mode_label.show()
            self.mode_combo.show()
        elif algorithm == "Euclidean Algorithm":
            # Hide rails, columns, and generate key
            self.rails_label.hide()
            self.rails_input.hide()
            self.columns_label.hide()
            self.columns_input.hide()
            self.generate_key_button.hide()
            self.mode_label.hide()
            self.mode_combo.hide()
            # Repurpose the key input for the first number
            self.key_label.setText("Enter first number (a):")
            self.key_label.show()
            self.key_input.show()
            # Repurpose plaintext label for the second number
            self.plaintext_label.setText("Enter second number (b):")
            # Change the process button text
            self.process_button.setText("Calculate GCD")
        elif algorithm == "Extended Euclidean Algorithm":
            # Hide rails, columns, and generate key
            self.rails_label.hide()
            self.rails_input.hide()
            self.columns_label.hide()
            self.columns_input.hide()
            self.generate_key_button.hide()
            self.mode_label.hide()
            self.mode_combo.hide()
             # Repurpose the key input for the first number
            self.key_label.setText("Enter first number (a):")
            self.key_label.show()
            self.key_input.show()
            # Repurpose plaintext label for the second number
            self.plaintext_label.setText("Enter second number (b):")
            # Change the process button text
            self.process_button.setText("Calculate")
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
            # Reset labels to default
            self.plaintext_label.setText("Plaintext:")
            self.process_button.setText("Process")

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
        elif algorithm == "One-Time Pad":
            text = self.plaintext_input.toPlainText()
            if not text:
                QMessageBox.warning(self, "Error", "Please enter some text to generate a key!")
                return
            random_key = ''.join(chr(random.randint(0, 255)) for _ in range(len(text)))
            self.key_input.setText(random_key)
        elif algorithm == "Simplified DES":
            # Generate a random 10-bit key
            random_key = ''.join(str(random.randint(0, 1)) for _ in range(10))
            self.key_input.setText(random_key)
        elif algorithm == "DES":
            # Generate a random 64-bit key
            random_key = ''.join(str(random.randint(0, 1)) for _ in range(64))
            self.key_input.setText(random_key)
        elif algorithm == "AES":
            # Generate a random 128-bit key
            random_key = ''.join(str(random.randint(0, 1)) for _ in range(128))
            self.key_input.setText(random_key)
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
            elif algorithm == "Hill Cipher":
                key = self.key_input.text()
                result = self.hill_cipher(text, key, mode)
            elif algorithm == "Vigenere Cipher":
                key = self.key_input.text()
                result = self.vigenere_cipher(text, key, mode)
            elif algorithm == "Caesar Cipher":
                shift = self.key_input.text()
                result = self.caesar_cipher(text, shift, mode)
            elif algorithm == "One-Time Pad":
                key = self.key_input.text()
                result = self.otp_cipher(text, key, mode)
            elif algorithm == "Simplified DES":
                key = self.key_input.text()
                result = self.simplified_des(text, key, mode)
            elif algorithm == "DES":
                key = self.key_input.text()

                if not all(bit in '01' for bit in key):
                    key = ''.join(format(ord(char), '04b') for char in key)

                if len(key) < 64:
                    key = key.ljust(64, '0')
                elif len(key) > 64:
                    key = key[:64]
                
                result = self.des(text, key, mode)
            elif algorithm == "AES":
                key = self.key_input.text()

                if not all(bit in '01' for bit in key):
                    key = ''.join(format(ord(char), '08b') for char in key)

                if len(key) < 128:
                    key = key.ljust(128, '0')
                elif len(key) > 128:
                    key = key[:128]
                
                result = self.aes(text, key, mode)
            elif algorithm == "Euclidean Algorithm":
                a = int(text)
                b = int(self.key_input.text())
                result = self.euclidean_algorithm(a, b)
            elif algorithm == "Extended Euclidean Algorithm":
                a = int(text)
                b = int(self.key_input.text())
                result = self.extended_euclidean_algorithm(a, b)
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

    def hill_cipher(self, text, key, mode):
        """Hill Cipher"""
        def create_matrix(key):
            key = key.upper().replace(" ", "")
            key_matrix = []
            size = int(len(key) ** 0.5)
            for i in range(size):
                row = [ord(char) - ord('A') for char in key[i*size:(i+1)*size]]
                key_matrix.append(row)
            return np.array(key_matrix)

        def mod_inverse(matrix, modulus):
            det = int(np.round(np.linalg.det(matrix)))
            det_inv = pow(det, -1, modulus)
            matrix_mod_inv = det_inv * np.round(det * np.linalg.inv(matrix)).astype(int) % modulus
            return matrix_mod_inv

        key_matrix = create_matrix(key)
        if mode == "Encrypt":
            text = text.upper().replace(" ", "")
            size = key_matrix.shape[0]
            text_vector = [ord(char) - ord('A') for char in text]
            while len(text_vector) % size != 0:
                text_vector.append(ord('X') - ord('A'))
            text_matrix = np.array(text_vector).reshape(-1, size)
            encrypted_matrix = (text_matrix @ key_matrix) % 26
            encrypted_text = ''.join(chr(num + ord('A')) for num in encrypted_matrix.flatten())
            return encrypted_text
        else:
            inv_key_matrix = mod_inverse(key_matrix, 26)
            text = text.upper().replace(" ", "")
            size = key_matrix.shape[0]
            text_vector = [ord(char) - ord('A') for char in text]
            text_matrix = np.array(text_vector).reshape(-1, size)
            decrypted_matrix = (text_matrix @ inv_key_matrix) % 26
            decrypted_text = ''.join(chr(num + ord('A')) for num in decrypted_matrix.flatten())
            return decrypted_text

    def vigenere_cipher(self, text, key, mode):
        """Vigenere Cipher"""
        def extend_key(text, key):
            key = key.upper().replace(" ", "")
            key = list(key)
            if len(text) == len(key):
                return ''.join(key)
            else:
                for i in range(len(text) - len(key)):
                    key.append(key[i % len(key)])
            return ''.join(key)

        key = extend_key(text, key)
        text = text.upper().replace(" ", "")
        if mode == "Encrypt":
            encrypted_text = []
            for i in range(len(text)):
                x = (ord(text[i]) + ord(key[i])) % 26
                x += ord('A')
                encrypted_text.append(chr(x))
            return ''.join(encrypted_text)
        else:
            decrypted_text = []
            for i in range(len(text)):
                x = (ord(text[i]) - ord(key[i]) + 26) % 26
                x += ord('A')
                decrypted_text.append(chr(x))
            return ''.join(decrypted_text)

    def caesar_cipher(self, text, shift, mode):
        """Caesar Cipher"""
        result = []
        shift = int(shift)
        if mode == "Decrypt":
            shift = -shift

        for char in text:
            if char.isalpha():
                shift_base = ord('A') if char.isupper() else ord('a')
                result.append(chr((ord(char) - shift_base + shift) % 26 + shift_base))
            else:
                result.append(char)

        return ''.join(result)

    def otp_cipher(self, text, key, mode):
        """One-Time Pad Cipher"""
        if len(key) < len(text):
            raise ValueError("Key must be at least as long as the text.")
        
        result = []
        for i in range(len(text)):
            if mode == "Encrypt":
                result.append(chr((ord(text[i]) + ord(key[i])) % 256))
            else:
                result.append(chr((ord(text[i]) - ord(key[i])) % 256))
        
        return ''.join(result)
    
    def simplified_des(self, text, key, mode):
        """Simplified DES (Data Encryption Standard)"""
        # Check if key is valid (10 bits)
        if not all(bit in '01' for bit in key) or len(key) != 10:
            raise ValueError("Key must be a 10-bit binary string (only 0s and 1s)")
        
        # Initial Permutation (IP)
        IP = [2, 6, 3, 1, 4, 8, 5, 7]
        # Final Permutation (FP) - inverse of IP
        FP = [4, 1, 3, 5, 7, 2, 8, 6]
        # Expansion Permutation (EP)
        EP = [4, 1, 2, 3, 2, 3, 4, 1]
        # P4 Permutation
        P4 = [2, 4, 3, 1]
        # P10 Permutation
        P10 = [3,5,2,7,4,10,1,9,8,6]
        # P8 Permutation
        P8 = [6, 3, 7, 4, 8, 5, 10, 9]
        
        # S-boxes
        S0 = [
            [1, 0, 3, 2],
            [3, 2, 1, 0],
            [0, 2, 1, 3],
            [3, 1, 3, 2]
        ]
        
        S1 = [
            [0, 1, 2, 3],
            [2, 0, 1, 3],
            [3, 0, 1, 0],
            [2, 1, 0, 3]
        ]
        
        def permute(bits, perm_table):
            return ''.join(bits[i-1] for i in perm_table)
        
        def shift_left(bits, n):
            return bits[n:] + bits[:n]
        
        def generate_subkeys(key):
            # Apply P10 permutation to the key
            key = permute(key, P10)
            
            # Split the key into two halves
            left = key[:5]
            right = key[5:]
            
            # Generate subkey K1 (shift both halves left by 1, then apply P8)
            left = shift_left(left, 1)
            right = shift_left(right, 1)
            k1 = permute(left + right, P8)
            
            # Generate subkey K2 (shift both halves left by 2, then apply P8)
            left = shift_left(left, 2)
            right = shift_left(right, 2)
            k2 = permute(left + right, P8)
            
            return k1, k2
        
        def f_function(right, subkey):
            # Expand right half from 4 to 8 bits
            expanded = permute(right, EP)
            
            # XOR with the subkey
            xored = ''.join(str(int(a) ^ int(b)) for a, b in zip(expanded, subkey))
            
            # Split into two 4-bit parts
            left = xored[:4]
            right = xored[4:]
            
            # Apply S-boxes
            row_l = int(left[0] + left[3], 2)
            col_l = int(left[1] + left[2], 2)
            s0_output = format(S0[row_l][col_l], '02b')
            
            row_r = int(right[0] + right[3], 2)
            col_r = int(right[1] + right[2], 2)
            s1_output = format(S1[row_r][col_r], '02b')
            
            # Combine and apply P4 permutation
            return permute(s0_output + s1_output, P4)
        
        def encrypt_block(block, k1, k2):
            # Initial permutation
            block = permute(block, IP)
            
            # Split into two 4-bit halves
            left = block[:4]
            right = block[4:]
            
            # First round
            new_right = ''.join(str(int(a) ^ int(b)) for a, b in zip(left, f_function(right, k1)))
            new_left = right
            
            # Second round
            left = new_left
            right = new_right
            new_right = ''.join(str(int(a) ^ int(b)) for a, b in zip(left, f_function(right, k2)))
            new_left = right
            
            # Combine and apply final permutation
            return permute(new_right + new_left, FP)
        
        def decrypt_block(block, k1, k2):
            # For decryption, use the subkeys in reverse order
            return encrypt_block(block, k2, k1)
        
        # Convert text to binary
        binary_text = ''.join(format(ord(char), '08b') for char in text)
        
        # Pad if necessary to make length a multiple of 8
        if len(binary_text) % 8 != 0:
            binary_text += '0' * (8 - (len(binary_text) % 8))
        
        # Generate subkeys
        k1, k2 = generate_subkeys(key)
        
        # Process each 8-bit block
        result = ""
        for i in range(0, len(binary_text), 8):
            block = binary_text[i:i+8]
            if mode == "Encrypt":
                processed_block = encrypt_block(block, k1, k2)
            else:
                processed_block = decrypt_block(block, k1, k2)
            
            # Convert binary back to character
            result += chr(int(processed_block, 2))
        
        return result
    def euclidean_algorithm(self, a, b):
        """Implementation of the Euclidean Algorithm to find the greatest common divisor (GCD)."""
        # Ensure a >= b
        if a < b:
            a, b = b, a
            
        # Keep track of the steps for display purposes
        steps = []
        
        while b != 0:
            steps.append(f"{a} = {b} × {a // b} + {a % b}")
            a, b = b, a % b
            
        # Format the output with steps and result
        result = "\n".join(steps)
        result += f"\n\nGCD = {a}"
        
        return result
        
    def extended_euclidean_algorithm(self, a, b):
        """Implementation of the Extended Euclidean Algorithm to find GCD and Bézout coefficients."""
        # Ensure a >= b
        swap = False
        if a < b:
            a, b = b, a
            swap = True
            
        # Initialize variables
        s0, s1 = 1, 0
        t0, t1 = 0, 1
        r0, r1 = a, b
        
        # Keep track of the steps for display purposes
        steps = []
        steps.append(f"Step 0: r0 = {r0}, s0 = {s0}, t0 = {t0}")
        steps.append(f"Step 1: r1 = {r1}, s1 = {s1}, t1 = {t1}")
        
        # Extended Euclidean Algorithm
        i = 1
        while r1 != 0:
            q = r0 // r1
            r0, r1 = r1, r0 - q * r1
            s0, s1 = s1, s0 - q * s1
            t0, t1 = t1, t0 - q * t1
            i += 1
            steps.append(f"Step {i}: q = {q}, r{i} = {r0}, s{i} = {s0}, t{i} = {t0}")
        
        # Format the output with steps and result
        result = "\n".join(steps)
        
        # Adjusting coefficients if we swapped a and b initially
        if swap:
            s0, t0 = t0, s0
            
        result += f"\n\nGCD({a}, {b}) = {r0}"
        result += f"\n{r0} = {s0}({a}) + {t0}({b})"
        
        # Addition for modular inverse
        if r0 == 1:  # If gcd is 1, modular inverse exists
            result += f"\n\nModular Inverse:"
            if swap:
                result += f"\n{a}⁻¹ mod {b} = {t0 % b}"
            else:
                result += f"\n{b}⁻¹ mod {a} = {s0 % a}"
        
        return result    
    def des(self, text, key, mode):

        ##596F7572206C6970732061726520736D6F6F74686572207468616E20766173656C696E650D0A0000

        #if not all(bit in '01' for bit in key) or len(key) != 64:
        #    raise ValueError("Key must be a 64-bit binary string (only 0s and 1s)")

        pc1 = [
            57, 49, 41, 33, 25, 17, 9,
            1, 58, 50, 42, 34, 26, 18,
            10, 2, 59, 51, 43, 35, 27,
            19, 11, 3, 60, 52, 44, 36,
            63, 55, 47, 39, 31, 23, 15,
            7, 62, 54, 46, 38, 30, 22,
            14, 6, 61, 53, 45, 37, 29,
            21, 13, 5, 28, 20, 12, 4
        ]

        pc2 = [
            14, 17, 11, 24, 1, 5,
            3, 28, 15, 6, 21, 10,
            23, 19, 12, 4, 26, 8,
            16, 7, 27, 20, 13, 2,
            41, 52, 31, 37, 47, 55,
            30, 40, 51, 45, 33, 48,
            44, 49, 39, 56, 34, 53,
            46, 42, 50, 36, 29, 32
        ]       

        ip = [
            58, 50, 42, 34, 26, 18, 10, 2,
            60, 52, 44, 36, 28, 20, 12, 4,
            62, 54, 46, 38, 30, 22, 14, 6,
            64, 56, 48, 40, 32, 24, 16, 8,
            57, 49, 41, 33, 25, 17, 9, 1,
            59, 51, 43, 35, 27, 19, 11, 3,
            61, 53, 45, 37, 29, 21, 13, 5,
            63, 55, 47, 39, 31, 23, 15, 7
        ]

        E = [
            32, 1, 2, 3, 4, 5,
            4, 5, 6, 7, 8, 9,
            8, 9, 10, 11, 12, 13,
            12, 13, 14, 15, 16, 17,
            16, 17, 18, 19, 20, 21,
            20, 21, 22, 23, 24, 25,
            24, 25, 26, 27, 28, 29,
            28, 29, 30, 31, 32, 1
        ]

        s_boxes = [
            # S1
            [
                [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
                [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
                [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
                [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
            ],
            # S2
            [
                [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
                [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
                [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
                [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
            ],
            # S3
            [
                [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
                [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
                [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
                [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
            ],
            # S4
            [
                [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
                [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
                [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
                [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
            ],
            # S5
            [
                [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
                [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
                [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
                [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
            ],
            # S6
            [
                [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
                [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
                [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
                [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
            ],
            # S7
            [
                [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
                [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
                [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
                [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
            ],
            # S8
            [
                [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
                [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
                [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
                [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
            ]
        ]

        P = [
            16, 7, 20, 21,
            29, 12, 28, 17,
            1, 15, 23, 26,
            5, 18, 31, 10,
            2, 8, 24, 14,
            32, 27, 3, 9,
            19, 13, 30, 6,
            22, 11, 4, 25
        ]

        P_inv = [
            40, 8, 48, 16, 56, 24, 64, 32,
            39, 7, 47, 15, 55, 23, 63, 31,
            38, 6, 46, 14, 54, 22, 62, 30,
            37, 5, 45, 13, 53, 21, 61, 29,
            36, 4, 44, 12, 52, 20, 60, 28,
            35, 3, 43, 11, 51, 19, 59, 27,
            34, 2, 42, 10, 50, 18, 58, 26,
            33, 1, 41, 9, 49, 17, 57, 25
        ]




        index = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]


        def permute(bits, perm_table):
            return ''.join(bits[i-1] for i in perm_table)

        def shift_left(bits, n):
            return bits[n:] + bits[:n]

        def generate_subkeys(key):
            key_56 = permute(key, pc1)

            left = key_56[:28]
            right = key_56[28:]

            keys = []
            for x in range (16) :
                left = shift_left(left, index[x])
                right = shift_left(right, index[x])
                keys.append(permute(left + right, pc2))

            return keys

        def f_function(right, subkey):
            expanded = permute(right, E)
            
            xored = ''.join(str(int(a) ^ int(b)) for a, b in zip(expanded, subkey))

            output = ''
            for i in range(8):
                block = xored[i*6:(i+1)*6]
            
                row = int(block[0] + block[5], 2)
                col = int(block[1:5], 2)

                s_output = format(s_boxes[i][row][col], '04b')
                output += s_output
            
            return permute(output, P)


        def encrypt_block(block, keys):
            block = permute(block, ip)

            left = block[:32]
            right = block[32:]
            
            for x in range(16):
                new_right = ''.join(str(int(a) ^ int(b)) for a, b in zip(left, f_function(right, keys[x])))
                left = right
                right = new_right

            return permute(right + left, P_inv)
        
        def decrypt_block(block, keys):
            return encrypt_block(block, keys[::-1])

        # Convert text to binary
        binary_text = ''.join(format(ord(char), '08b') for char in text)
        
        # Pad if necessary to make length a multiple of 8
        if len(binary_text) % 8 != 0:
            binary_text += '0' * (8 - (len(binary_text) % 8))
        
        # Generate subkeys
        keys = generate_subkeys(key)

        # Process each 64-bit block
        result = ""
        for i in range(0, len(binary_text), 64):
            block = binary_text[i:i+64]
            if len(block) < 64:
                block += '0' * (64 - len(block))

            if mode == "Encrypt":
                processed_block = encrypt_block(block, keys)
            else:
                processed_block = decrypt_block(block, keys)

            # Convert binary back to characters (8 bits per character)
            for j in range(0, 64, 8):
                if j+8 <= len(processed_block):
                    result += chr(int(processed_block[j:j+8], 2))
                    
        return result
    
    def aes(self, text, key, mode):
        """Advanced Encryption Standard (AES) implementation from scratch"""
        # AES S-box for SubBytes operation
        s_box = [
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
            0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
            0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
            0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
            0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
            0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
            0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
            0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
            0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
            0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
            0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
            0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
            0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
            0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
            0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
        ]
        
        # Inverse S-box for InvSubBytes operation
        inv_s_box = [
            0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
            0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
            0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
            0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
            0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
            0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
            0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
            0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
            0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
            0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
            0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
            0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
            0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
            0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
            0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
            0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
        ]
        
        # Rcon values for key expansion
        rcon = [
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
        ]
        
        # Convert text to a padded array of bytes
        def text_to_blocks(text):
            # Convert text to bytes
            blocks = []
            text_bytes = text.encode('utf-8')
            
            # Add padding: PKCS#7
            padding_len = 16 - (len(text_bytes) % 16)
            padded_text = text_bytes + bytes([padding_len] * padding_len)
            
            # Create blocks of 16 bytes (128 bits)
            for i in range(0, len(padded_text), 16):
                block = padded_text[i:i+16]
                blocks.append(block)
            
            return blocks
        
        # Convert blocks back to text and remove padding
        def blocks_to_text(blocks):
            result = b''
            for block in blocks:
                result += bytes(block)
            
            # Remove PKCS#7 padding
            padding_len = result[-1]
            if padding_len < 16:
                return result[:-padding_len].decode('utf-8', errors='replace')
            else:
                return result.decode('utf-8', errors='replace')
        
        # Convert binary key to bytes
        def binary_to_key(binary_key):
            key_bytes = bytearray(16)  # 16 bytes = 128 bits
            for i in range(16):
                byte_val = 0
                for j in range(8):
                    bit_index = i * 8 + j
                    if bit_index < len(binary_key) and binary_key[bit_index] == '1':
                        byte_val |= (1 << (7 - j))
                key_bytes[i] = byte_val
            return key_bytes
        
        # Key expansion to generate round keys
        def key_expansion(key):
            # Convert binary key string to actual key bytes
            if all(bit in '01' for bit in key):
                key = binary_to_key(key)
            else:
                # If key is not binary, treat as ASCII and pad/truncate to 16 bytes
                key = key.encode('utf-8')
                if len(key) < 16:
                    key = key.ljust(16, b'\x00')
                else:
                    key = key[:16]
            
            # Key expansion: 10 rounds for AES-128 -> 11 round keys (including initial)
            round_keys = [bytearray(16) for _ in range(11)]
            round_keys[0] = bytearray(key)
            
            for i in range(1, 11):
                # Take last 4 bytes of previous round key
                temp = bytearray(round_keys[i-1][-4:])
                
                # Rotate word
                temp = temp[1:] + temp[:1]
                
                # Apply S-box
                for j in range(4):
                    temp[j] = s_box[temp[j]]
                
                # XOR with round constant (only first byte)
                temp[0] ^= rcon[i-1]
                
                # Generate new round key
                for j in range(4):
                    for k in range(4):
                        idx = j * 4 + k
                        prev_idx = idx - 16 if idx >= 16 else idx
                        round_keys[i][idx] = round_keys[i-1][idx] ^ (temp[k] if j == 0 else round_keys[i][prev_idx])
            
            return round_keys
        
        # SubBytes transformation
        def sub_bytes(state):
            for i in range(16):
                state[i] = s_box[state[i]]
            return state
        
        # InvSubBytes transformation
        def inv_sub_bytes(state):
            for i in range(16):
                state[i] = inv_s_box[state[i]]
            return state
        
        # ShiftRows transformation
        def shift_rows(state):
            # Convert to 4x4 matrix form for easier shifting
            matrix = [state[i:i+4] for i in range(0, 16, 4)]
            
            # Shift rows (1st row not shifted)
            matrix[1] = matrix[1][1:] + matrix[1][:1]  # Shift 1 to left
            matrix[2] = matrix[2][2:] + matrix[2][:2]  # Shift 2 to left
            matrix[3] = matrix[3][3:] + matrix[3][:3]  # Shift 3 to left
            
            # Convert back to flat array
            return bytearray([matrix[i//4][i%4] for i in range(16)])
        
        # InvShiftRows transformation
        def inv_shift_rows(state):
            # Convert to 4x4 matrix form
            matrix = [state[i:i+4] for i in range(0, 16, 4)]
            
            # Inverse shift rows (1st row not shifted)
            matrix[1] = matrix[1][3:] + matrix[1][:3]  # Shift 1 to right
            matrix[2] = matrix[2][2:] + matrix[2][:2]  # Shift 2 to right
            matrix[3] = matrix[3][1:] + matrix[3][:1]  # Shift 3 to right
            
            # Convert back to flat array
            return bytearray([matrix[i//4][i%4] for i in range(16)])
        
        # Helper function for MixColumns
        def galois_multiply(a, b):
            p = 0
            for i in range(8):
                if b & 1:
                    p ^= a
                high_bit_set = a & 0x80
                a <<= 1
                if high_bit_set:
                    a ^= 0x1B  # XOR with irreducible polynomial x^8 + x^4 + x^3 + x + 1
                b >>= 1
            return p & 0xFF
        
        # MixColumns transformation
        def mix_columns(state):
            result = bytearray(16)
            
            for i in range(0, 16, 4):
                result[i] = galois_multiply(state[i], 2) ^ galois_multiply(state[i+1], 3) ^ state[i+2] ^ state[i+3]
                result[i+1] = state[i] ^ galois_multiply(state[i+1], 2) ^ galois_multiply(state[i+2], 3) ^ state[i+3]
                result[i+2] = state[i] ^ state[i+1] ^ galois_multiply(state[i+2], 2) ^ galois_multiply(state[i+3], 3)
                result[i+3] = galois_multiply(state[i], 3) ^ state[i+1] ^ state[i+2] ^ galois_multiply(state[i+3], 2)
            
            return result
        
        # InvMixColumns transformation
        def inv_mix_columns(state):
            result = bytearray(16)
            
            for i in range(0, 16, 4):
                result[i] = galois_multiply(state[i], 0x0E) ^ galois_multiply(state[i+1], 0x0B) ^ galois_multiply(state[i+2], 0x0D) ^ galois_multiply(state[i+3], 0x09)
                result[i+1] = galois_multiply(state[i], 0x09) ^ galois_multiply(state[i+1], 0x0E) ^ galois_multiply(state[i+2], 0x0B) ^ galois_multiply(state[i+3], 0x0D)
                result[i+2] = galois_multiply(state[i], 0x0D) ^ galois_multiply(state[i+1], 0x09) ^ galois_multiply(state[i+2], 0x0E) ^ galois_multiply(state[i+3], 0x0B)
                result[i+3] = galois_multiply(state[i], 0x0B) ^ galois_multiply(state[i+1], 0x0D) ^ galois_multiply(state[i+2], 0x09) ^ galois_multiply(state[i+3], 0x0E)
            
            return result
        
        # AddRoundKey transformation
        def add_round_key(state, round_key):
            for i in range(16):
                state[i] ^= round_key[i]
            return state
        
        # AES encryption of a single block
        def encrypt_block(block, round_keys):
            state = bytearray(block)
            
            # Initial round
            state = add_round_key(state, round_keys[0])
            
            # Main rounds
            for i in range(1, 10):
                state = sub_bytes(state)
                state = shift_rows(state)
                state = mix_columns(state)
                state = add_round_key(state, round_keys[i])
            
            # Final round (no MixColumns)
            state = sub_bytes(state)
            state = shift_rows(state)
            state = add_round_key(state, round_keys[10])
            
            return state
        
        # AES decryption of a single block
        def decrypt_block(block, round_keys):
            state = bytearray(block)
            
            # Initial round
            state = add_round_key(state, round_keys[10])
            state = inv_shift_rows(state)
            state = inv_sub_bytes(state)
            
            # Main rounds
            for i in range(9, 0, -1):
                state = add_round_key(state, round_keys[i])
                state = inv_mix_columns(state)
                state = inv_shift_rows(state)
                state = inv_sub_bytes(state)
            
            # Final round
            state = add_round_key(state, round_keys[0])
            
            return state
        
        # Main AES function
        try:
            # Generate round keys
            round_keys = key_expansion(key)
            
            if mode == "Encrypt":
                # Convert text to blocks
                blocks = text_to_blocks(text)
                
                # Encrypt each block
                encrypted_blocks = [encrypt_block(block, round_keys) for block in blocks]
                
                # Convert to base64 for display
                encrypted_bytes = b''.join(bytes(block) for block in encrypted_blocks)
                return base64.b64encode(encrypted_bytes).decode('utf-8')
            else:
                # Decode base64 input
                try:
                    encrypted_bytes = base64.b64decode(text)
                except:
                    # If not base64, treat as raw encrypted bytes
                    encrypted_bytes = text.encode('utf-8')
                
                # Split into blocks
                blocks = [encrypted_bytes[i:i+16] for i in range(0, len(encrypted_bytes), 16)]
                
                # Decrypt each block
                decrypted_blocks = [decrypt_block(block, round_keys) for block in blocks]
                
                # Convert blocks back to text
                return blocks_to_text(decrypted_blocks)
        except Exception as e:
            return f"AES Error: {str(e)}"
        








if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EncryptionApp()
    app.setWindowIcon(QIcon("encryptiontoolLogo.icns"))
    app.setApplicationName("Encryption Tool")
    window.setWindowIcon(QIcon("encryptiontoolLogo.icns"))    
    window.setWindowTitle("Encryption Tool")
    window.show()
    sys.exit(app.exec())

            

            





        








if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EncryptionApp()
    app.setWindowIcon(QIcon("encryptiontoolLogo.icns"))
    app.setApplicationName("Encryption Tool")
    window.setWindowIcon(QIcon("encryptiontoolLogo.icns"))    
    window.setWindowTitle("Encryption Tool")
    window.show()
    sys.exit(app.exec())
