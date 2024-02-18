$PYTHON='python3.9'

# Remove old virtual enviroment
if (Test-Path ".\env") {
  Write-Host "Virutal enviroment exists. Deleting!"
  Remove-Item -Recurse -Force .\env
}

# Make new one
Write-Host "Creating Virtual enviroment at .\env and activating"
python -m venv env
.\env\Scripts\Activate
python -m pip install -U pip
python -m pip install wheel

if (Test-Path ".\requirements.txt") {
  Write-Host "Found requirements.txt, installing"
  python -m pip install -r .\requirements.txt
}