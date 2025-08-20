# 🛡️Comment Toxicity Detection System

A comprehensive deep learning solution for detecting and classifying toxic comments across multiple categories using Natural Language Processing and LSTM neural networks.

## 🎯 Overview

Online toxicity is a growing concern across digital platforms. This project provides an end-to-end solution for automatically detecting and classifying toxic comments into six categories: **toxic**, **severe_toxic**, **obscene**, **threat**, **insult** and **identity_hate**.

### Key Highlights
- **Multi-label Classification** - Comments can belong to multiple toxic categories simultaneously.
- **Advanced NLP Pipeline** - Comprehensive text preprocessing with lemmatization and stopword removal.
- **Deep Learning Architecture** - Bidirectional LSTM with embedding layers for superior performance.
- **Interactive Web Interface** - Real-time predictions through a user-friendly Streamlit application.
- **Comprehensive Analysis** - Visual insights including word clouds and performance metrics.

## ✨ Features

### Core Functionality
- ✅ **Multi-label toxicity classification** across 6 categories.
- ✅ **Robust text preprocessing** with cleaning, lemmatization and tokenization.
- ✅ **Dual feature engineering** using TF-IDF vectorization and sequence embeddings.
- ✅ **Bidirectional LSTM model** with dropout regularization.
- ✅ **Real-time and batch predictions** through interactive web interface.

### Visualization & Analytics
- ✅ **Label distribution analysis** with interactive charts.
- ✅ **Comment length distribution** visualization.
- ✅ **Word clouds** for each toxicity category.
- ✅ **Model performance tracking** with accuracy and loss plots.
- ✅ **Interactive prediction results** with probability scores.

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn, wordcloud |
| **NLP** | NLTK, scikit-learn |
| **Deep Learning** | TensorFlow/Keras |
| **Web Framework** | Streamlit |
| **Deployment** | pyngrok |

## 📊 Dataset

The project uses the Toxic Comment Classification dataset with the following structure - 

### Input Data
- **File Format** - CSV (Train.csv, Test.csv)
- **Primary Feature** - `comment_text`(Raw user comments)
- **Target Labels** - 6 binary toxicity categories

### Toxicity Categories
1. **toxic** - General toxicity.
2. **severe_toxic** - Extremely toxic content.
3. **obscene** - Obscene language.
4. **threat** - Threatening behavior.
5. **insult** - Personal insults.
6. **identity_hate** - Identity-based harassment.

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow streamlit pyngrok wordcloud
```

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/iamhriturajsaha/COMMENT-TOXICITY-DETECTION-SYSTEM.git
   cd COMMENT-TOXICITY-DETECTION-SYSTEM
   ```

2. **Prepare your data**
   - Place `Train.csv` and `Test.csv` in the project directory
   - Ensure CSV files contain the required columns

3. **Run the training pipeline**
   ```bash
   # Open Comment.ipynb in Jupyter/Colab and execute all cells
   # This will generate the trained model and preprocessing artifacts
   ```

4. **Launch the web application**
   ```bash
   streamlit run app.py
   ```

## 🔧 Project Architecture

### 1. Data Preprocessing Pipeline
```
Raw Text → Lowercase → Remove URLs/HTML → Tokenization → 
Remove Stopwords → Lemmatization → Clean Text
```

### 2. Feature Engineering
- **TF-IDF Vectorization** - Maximum 5,000 features
- **Sequence Tokenization** - Maximum 50,000 words, padded to 200 tokens

### 3. Model Architecture
```python
Sequential([
    Embedding(max_words=50000, output_dim=128),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    GlobalMaxPooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(6, activation="sigmoid")  # Multi-label output
])
```

### 4. Training Configuration
- **Loss Function**: Binary crossentropy (multi-label)
- **Optimizer**: Adam
- **Epochs**: 10
- **Batch Size**: 256
- **Validation Split**: 20%

## 💻 Usage

### Web Application Features

#### Single Comment Prediction
1. Enter a comment in the text area.
2. Click "Predict".
3. View probability scores for each category.
4. See visual representation via bar chart.

#### Batch Prediction
1. Upload a CSV file with comments.
2. Process multiple comments simultaneously.
3. Download results with toxicity classifications.
4. Analyze batch statistics.

#### Model Insights
- View training history (accuracy/loss curves).
- Explore data visualizations.
- Understand model performance metrics.

## 📈 Model Performance

The model achieves robust performance across all toxicity categories with -
- Effective handling of class imbalance.
- Strong generalization to unseen data.
- Balanced precision-recall trade-offs.

## 🔮 Future Enhancements

- **Transformer Integration** - Implement BERT/RoBERTa for improved accuracy.
- **Model Explainability** - Add SHAP/LIME for prediction interpretability.
- **Enhanced UI** - Develop more interactive dashboards and visualizations.
- **Production Deployment** - Migrate to cloud platforms (AWS, HuggingFace Spaces, Heroku).
- **Real-time Processing** - Implement streaming data pipeline.
- **Multi-language Support** - Extend to non-English comment detection.