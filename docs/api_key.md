# API Key setup

1. Create `ke.env` in repository root with:
```
GOOGLE_API_KEY=your-key
```
2. Ensure `.gitignore` includes `ke.env` and `.env`
3. Verify key loaded by running:
```bash
python -c "import os; print(os.getenv('GOOGLE_API_KEY'))"
```
4. If not set, run in shell:
```bash
setx GOOGLE_API_KEY "your-key"  # Windows
export GOOGLE_API_KEY="your-key"  # Linux/macOS
```
