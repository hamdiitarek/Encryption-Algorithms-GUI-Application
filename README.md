# 🔐 Encryption Algorithms GUI Application

A comprehensive desktop application built with Python and PySide6 that provides implementations of various classic and modern encryption algorithms for educational purposes.

## Features

### � **16 Cryptographic Algorithms**
- **Classical Ciphers:** Rail Fence, Route, Playfair, Hill, Vigenère, Caesar, One-Time Pad
- **Modern Encryption:** Simplified DES, Full DES, AES (Advanced Encryption Standard)
- **Hash Functions:** MD5, SHA-1, SHA-256, SHA-512
- **Mathematical Tools:** Euclidean Algorithm, Extended Euclidean Algorithm

### **Modern User Interface**
- Clean, intuitive GUI built with PySide6/Qt6
- Dark/Light theme toggle
- Dynamic interface that adapts to selected algorithm
- Real-time encryption/decryption processing
- Character frequency analysis with matplotlib

### **Smart Features**
- Automatic key generation for supported algorithms
- Input validation and comprehensive error handling
- Educational warnings and algorithm explanations
- Cross-platform compatibility (Windows, macOS, Linux)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/Encryption-Algorithms-GUI-Application.git
cd Encryption-Algorithms-GUI-Application

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Requirements

- **Python 3.8+**
- **PySide6** - Modern Qt6-based GUI framework
- **matplotlib** - For frequency analysis visualization
- **numpy** - Numerical computations
- **requests** - API calls for key generation

## Usage Examples

### Caesar Cipher
```
Input: "HELLO WORLD"
Shift: 3
Output: "KHOOR ZRUOG"
```

### AES Encryption
```
Plaintext: "Secret Message"
Key: 128-bit binary key
Mode: Encrypt/Decrypt
```

### Hash Functions
```
Input: "password123"
MD5: 482c811da5d5b4bc6d497ffa98491e38
SHA-256: ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f
```

## Project Structure

```
Encryption-Algorithms-GUI-Application/
├── main.py                    # Main application
├── requirements.txt           # Dependencies
├── encryptiontoolLogo.icns   # App icon
├── EncryptionTool.spec       # Build configuration
├── build.md                  # Build instructions
└── README.md                 # Documentation
```

## Building Executables

### For macOS:
```bash
pip install pyinstaller
pyinstaller EncryptionTool.spec
```

### For Windows/Linux:
```bash
pyinstaller --windowed --name EncryptionTool --icon encryptiontoolLogo.icns main.py
```

## Educational Notice

This application is designed for **educational purposes only**. While algorithms follow standard specifications, this implementation should not be used for production security systems without proper security review.
