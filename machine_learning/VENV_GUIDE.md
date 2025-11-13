# Virtual Environment Guide

## ✅ Virtual Environment Created

A Python virtual environment has been set up in the `machine_learning` folder with all required dependencies installed.

---

## 🚀 How to Use

### Activate the Virtual Environment

**PowerShell:**
```powershell
cd machine_learning
.\venv\Scripts\Activate.ps1
```

**Command Prompt:**
```cmd
cd machine_learning
venv\Scripts\activate.bat
```

**Git Bash / Linux / macOS:**
```bash
cd machine_learning
source venv/bin/activate
```

### Verify Activation

When activated, you should see `(venv)` at the beginning of your terminal prompt:
```
(venv) PS C:\...\machine_learning>
```

### Check Installed Packages

```powershell
pip list
```

You should see:
- numpy==1.24.3
- pandas==2.0.3
- scipy==1.11.1

---

## 📊 Run the Synthetic Data Generator

### Generate Default Dataset

```powershell
# Make sure virtual environment is activated first!
cd data
python synthetic_data_generator.py
```

### Generate Custom Dataset

```powershell
python synthetic_data_generator.py --num-transformers 100 --num-meters 5000
```

### Run Examples

```powershell
python examples.py
```

### Run Tests

```powershell
# Install pytest first
pip install pytest

# Run tests
pytest test_generator.py -v
```

---

## 🛑 Deactivate Virtual Environment

When you're done working:
```powershell
deactivate
```

---

## 🔄 Reactivate Later

Every time you start a new terminal session, you'll need to activate the virtual environment again:

```powershell
cd "c:\Users\Ken Ira Talingting\Desktop\GhostLoadMapper-IDOL_Hackathon-\machine_learning"
.\venv\Scripts\Activate.ps1
```

---

## 📦 Install Additional Packages

If you need to install more packages:

```powershell
# Make sure venv is activated
pip install package-name

# Or install from requirements file
pip install -r requirements.txt
```

---

## 🗑️ Delete Virtual Environment

If you ever need to recreate it:

```powershell
# Deactivate first
deactivate

# Delete the folder
Remove-Item -Recurse -Force venv

# Recreate
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r data\requirements.txt
```

---

## ✅ Quick Test

Verify everything works:

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Test imports
python -c "import numpy; import pandas; import scipy; print('All packages installed successfully!')"

# Generate small dataset
cd data
python synthetic_data_generator.py --num-meters 100 --num-transformers 5 --output-dir test_output
```

---

## 📝 Notes

- ✅ Virtual environment is in `machine_learning/venv/`
- ✅ All dependencies installed: numpy, pandas, scipy
- ✅ Ready to run synthetic data generator
- ✅ Isolated from system Python packages
- ✅ Can safely install additional packages without affecting system

---

**Created**: November 13, 2025  
**Python Version**: 3.10  
**Status**: ✅ Ready to use
