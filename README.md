# AI Resume Screening & ATS Parsing Tool

A modern, production-ready AI Resume Screening and Applicant Tracking System (ATS) built with Streamlit and Python. The application parses candidate profiles, evaluates key qualifications, computes custom ATS compatibility scores based on job descriptions, compares candidates side-by-side using visualization, and automatically classifies job roles using a machine learning pipeline.

🚀 **Live App URL:** [https://resume-screen-phc3mgrkvgajj3v7m6chxy.streamlit.app/](https://resume-screen-phc3mgrkvgajj3v7m6chxy.streamlit.app/)

---

## ✨ Features

- **📊 Dashboard**: High-level overview metrics showing total candidates, average ATS compatibility scores, candidate grade levels, and category distribution pie charts.
- **📥 Upload & Analyze**: Upload multiple resumes (PDF, DOCX, TXT) at once. Optionally paste a Job Description (JD) to compute structured ATS matches. View detailed candidate cards showing extracted info, skill badges, education records, and years of experience.
- **⚔️ Candidate Comparison**: Select multiple applicants and compare their skillsets and ATS component scores side-by-side on an interactive Radar Chart. Find skill overlaps and unique candidate qualifications.
- **📈 Analytics**: Deep-dive visualizations illustrating the overall score distribution, grade tiers (Excellent, Good, Average, Poor), and most frequent skills found across all candidate records.
- **⚙️ Admin Panel**: Verify the loaded Machine Learning model health, review classification model training report details, clear application state, and export candidate evaluations to CSV.

---

## 🛠️ Tech Stack & Libraries

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (TF-IDF Vectorization, Label Encoding, Random Forest Classifier)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization**: [Plotly Express & Graph Objects](https://plotly.com/)
- **Resume Text Extraction**:
  - `pdfplumber` & `PyPDF2` (for PDF documents)
  - `python-docx` (for Word documents)
- **Styling**: Custom CSS for premium dark glassmorphism effects and custom animations.

---

## 📂 Project Structure

```
resume-screening-tool/
├── dataset/
│   └── resume_dataset.csv       # Training dataset containing categories and resumes
├── models/                      # Saved ML artifacts (vectorizer, classifier, encoder, reports)
│   ├── label_encoder.pkl
│   ├── resume_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── training_report.txt
├── uploads/                     # Temp storage directory for parsed files
├── app.py                       # Main Streamlit web application
├── parser.py                    # Text extraction (PDF, DOCX, TXT) and entity heuristic parsers
├── train_model.py               # Model training script evaluating SVC, RF, Logistic Regression, etc.
├── style.css                    # Custom CSS variables for app styling
├── script.js                    # Custom JavaScript animations and gauge logic
├── requirements.txt             # Python dependency packages
└── README.md                    # Project documentation
```

---

## 🚀 Running Locally

### 1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/NITHYA123-hub/resume-screening-tool.git
cd resume-screening-tool
```

### 2. Install all dependencies:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train the Machine Learning Model:
To run the automated model evaluation pipeline and re-save the classifier artifacts:
```bash
python train_model.py
```
*The script will automatically augment dataset records, evaluate Logistic Regression, SVC, Random Forest, Naive Bayes, and KNN models, and save the best-performing model (Random Forest, ~90% accuracy) in the `/models` directory.*

### 4. Start the Streamlit App:
```bash
streamlit run app.py
```
*Open `http://localhost:8501` in your browser.*

---

## 🤖 Machine Learning Details

The resume classification pipeline uses:
1. **Text Preprocessing**: Lowercasing, removing URLs, punctuation, and extra spaces.
2. **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer using uni-grams and bi-grams (`ngram_range=(1,2)`) and English stop word filtering.
3. **Classification**: Evaluated multiple classifiers using Stratified 5-Fold Cross-Validation. The **Random Forest Classifier** achieved the highest test accuracy of **90%** and was saved as the master predictor.
