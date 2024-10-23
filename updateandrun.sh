echo "pulling changes"
git pull
git status

echo "Checking pip"
pip install -r requirements.txt

echo "running..."
python -m main.py