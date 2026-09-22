# ABC Ltd. — Employee Attrition Predictor

A logistic regression tool that predicts an employee's probability of leaving the company, built for managerial decision support (Assignment 2: Predictive Analytics & Managerial AI Adoption).

## Contents

| File | Purpose |
|---|---|
| `ABC_Ltd_Employee_Attrition.csv` | Dataset (1,470 employees, 35 features) |
| `ABC_Ltd_Attrition_Predictor.ipynb` | Google Colab notebook — data prep, model training, evaluation |
| `attrition_model.pkl` | Trained logistic regression model |
| `scaler.pkl` | Feature scaler (must be applied before prediction) |
| `feature_columns.pkl` | Exact column order the model expects |
| `app.py` | Streamlit web app — the deployable tool for non-technical users |
| `requirements.txt` | Python dependencies for deployment |

## Model performance
- ROC-AUC ≈ 0.80
- Trained with `class_weight='balanced'` to handle the ~84/16 class imbalance (most employees don't leave)

---

## Part A — Run the notebook in Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook → select `ABC_Ltd_Attrition_Predictor.ipynb`
3. Run cells top to bottom; when prompted, upload `ABC_Ltd_Employee_Attrition.csv`
4. The last cells export `attrition_model.pkl`, `scaler.pkl`, and `feature_columns.pkl`, which you'll need for deployment

---

## Part B — Push everything to GitHub

### Step 1: Create a GitHub repository
1. Go to [github.com/new](https://github.com/new)
2. Name it something like `abc-ltd-attrition-predictor`
3. Set it to **Public** (required for free Streamlit Cloud deployment)
4. Click **Create repository**

### Step 2: Upload the files
**Option A — via the GitHub website (easiest, no terminal needed):**
1. On your new repo's page, click **"uploading an existing file"**
2. Drag in all files: `app.py`, `requirements.txt`, `attrition_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `ABC_Ltd_Employee_Attrition.csv`, `ABC_Ltd_Attrition_Predictor.ipynb`, `README.md`
3. Scroll down, add a commit message like "Initial commit", click **Commit changes**

**Option B — via terminal/git:**
```bash
git clone https://github.com/YOUR_USERNAME/abc-ltd-attrition-predictor.git
cd abc-ltd-attrition-predictor
# copy all the project files into this folder, then:
git add .
git commit -m "Initial commit: attrition predictor"
git push origin main
```

### Step 3: Verify
Refresh your repo page — you should see all files listed, including `app.py` and the `.pkl` model files.

---

## Part C — Deploy the tool live with Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account
2. Click **"New app"**
3. Select your repository (`abc-ltd-attrition-predictor`), branch (`main`), and main file path (`app.py`)
4. Click **Deploy**
5. Wait 1–2 minutes while it installs dependencies from `requirements.txt`
6. You'll get a public URL like `https://abc-ltd-attrition-predictor.streamlit.app` — this is the link you share with managers for your user study

### If deployment fails
- Check the app logs (shown on the Streamlit Cloud dashboard) for missing packages — add anything missing to `requirements.txt`
- Make sure `attrition_model.pkl`, `scaler.pkl`, and `feature_columns.pkl` were actually uploaded to the repo (not just left on your local machine)

---

## Part D — Using the tool for your managerial user study
1. Send the Streamlit URL to your managers/professionals/team leads
2. Ask them to enter 2–3 employee profiles (real, anonymized, or hypothetical) and note the predicted risk
3. Follow up with the qualitative questions from your assignment (trust, explainability, AI vs. experience, adoption barriers, etc.)
