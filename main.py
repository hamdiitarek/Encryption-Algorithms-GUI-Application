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
            "Simplified DES"  
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
        
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EncryptionApp()
    app.setWindowIcon(QIcon("encryptiontoolLogo.icns"))
    app.setApplicationName("Encryption Tool")
    window.setWindowIcon(QIcon("encryptiontoolLogo.icns"))    
    window.setWindowTitle("Encryption Tool")
    window.show()
    sys.exit(app.exec())
