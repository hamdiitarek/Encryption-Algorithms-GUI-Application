Creating an installable macOS application bundle (`.app`) for your PySide6-based encryption tool involves several steps. Below is a detailed guide to help you package your Python script into a macOS application.

---

### Steps to Create a macOS Installer

#### 1. **Prepare Project**
Project structure should look something like this:
```
EncryptionTool/
├── main.py              # Your main script
├── encryptiontoolLogo.icns  # Your application icon
├── requirements.txt     # Python dependencies
└── README.md            # Optional: Documentation
```

---

#### 2. **Install Required Tools**
You’ll need the following tools:
- **PyInstaller**: To package your Python script into an executable.
- **Create-dmg**: To create a `.dmg` file for distribution.

Install them using `pip` and `brew`:
```bash
pip install pyinstaller
brew install create-dmg
```

#### 3. **Package Application with PyInstaller**
Use PyInstaller to create a standalone macOS application bundle.

1. Create a `.spec` file for PyInstaller:
   ```bash
   pyinstaller --windowed --name EncryptionTool --icon encryptiontoolLogo.icns main.py
   ```

2. Edit the generated `EncryptionTool.spec` file to include additional resources (e.g., icons, data files):
   ```python
   # EncryptionTool.spec
   # -*- mode: python ; coding: utf-8 -*-

   block_cipher = None

   a = Analysis(
       ['main.py'],
       pathex=['/path/to/EncryptionTool'],
       binaries=[],
       datas=[('encryptiontoolLogo.icns', '.')],  # Include the icon file
       hiddenimports=[],
       hookspath=[],
       hooksconfig={},
       runtime_hooks=[],
       excludes=[],
       win_no_prefer_redirects=False,
       win_private_assemblies=False,
       cipher=block_cipher,
   )
   pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

   exe = EXE(
       pyz,
       a.scripts,
       [],
       exclude_binaries=True,
       name='EncryptionTool',
       debug=False,
       bootloader_ignore_signals=False,
       strip=False,
       upx=True,
       console=False,
       disable_windowed_traceback=False,
       target_arch=None,
       codesign_identity=None,
       entitlements_file=None,
   )
   coll = COLLECT(
       exe,
       a.binaries,
       a.zipfiles,
       a.datas,
       strip=False,
       upx=True,
       upx_exclude=[],
       name='EncryptionTool',
   )
   app = BUNDLE(
       coll,
       name='EncryptionTool.app',
       icon='encryptiontoolLogo.icns',
       bundle_identifier='com.yourapp.encryptiontool',
   )
   ```

3. Run PyInstaller with the `.spec` file:
   ```bash
   pyinstaller EncryptionTool.spec
   ```

This will create a `dist/EncryptionTool.app` bundle.

---

#### 4. **Test Application**
Navigate to the `dist` folder and double-click `EncryptionTool.app` to test it. Ensure everything works as expected.

---

#### 5. **Create a DMG File**
Use `create-dmg` to create a `.dmg` file for distribution.

1. Create a directory structure for the DMG:
   ```bash
   mkdir -p dmg/EncryptionTool
   cp -r dist/EncryptionTool.app dmg/EncryptionTool/
   ```

2. Run `create-dmg`:
   ```bash
   create-dmg \
       --volname "EncryptionTool" \
       --volicon "encryptiontoolLogo.icns" \
       --background "background.png" \  
       --window-pos 200 120 \
       --window-size 800 400 \
       --icon-size 100 \
       --icon "EncryptionTool.app" 200 190 \
       --hide-extension "EncryptionTool.app" \
       --app-drop-link 600 185 \
       "EncryptionTool.dmg" \
       "dmg/"
   ```

This will create a `EncryptionTool.dmg` file in the current directory.


### Example Directory Structure After Packaging
```
EncryptionTool/
├── main.py
├── encryptiontoolLogo.icns
├── EncryptionTool.spec
├── dist/
│   └── EncryptionTool.app
├── dmg/
│   └── EncryptionTool.app
└── EncryptionTool.dmg
```
