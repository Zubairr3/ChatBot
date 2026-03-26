# Installation

1. Clone repository
2. Setup virtual env:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Create `ke.env` with:
```
GOOGLE_API_KEY=your-key
```

5. Run sample:
```bash
python Rag_Char.py --ui
```
