echo off
echo Installing dependencies...
pip install -r requirements.txt
echo Running dataset preparation...
REM Ensure you have the correct Python environment activated
python prepare_dataset.py
echo Running training script...
REM Ensure you have the correct Python environment activated
python train.py
echo Training completed.
pause