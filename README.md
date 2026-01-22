# House Price Prediction System

A machine learning web application that predicts house prices using Random Forest Regressor based on property features.

## 📋 Project Structure

```
HousePrice_Project_MeneAnirejuoritse_23CG034095/
├── app.py                              # Flask web application
├── requirements.txt                     # Python dependencies                 # 
├── HousePrice_hosted_webGUI_link.txt   # Submission information
├── train.csv                           # Dataset (download from Kaggle)
├── model/
│   ├── model_building.ipynb            # Model development notebook
│   ├── house_price_model.pkl           # Trained model (generated)
│   ├── scaler.pkl                      # Feature scaler (generated)
│   ├── neighborhood_encoder.pkl        # Label encoder (generated)
│   └── feature_names.pkl               # Feature reference (generated)
├── static/
│   └── style.css                       # External stylesheet
└── templates/
    └── index.html                      # Web interface
```

**Note**: All `.pkl` files are generated when you run the model_buildingpython3 .

## 🚀 Features

- **6 Selected Features**: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, YearBuilt, Neighborhood
- **Algorithm**: Random Forest Regressor
- **Model Persistence**: Joblib
- **Web Framework**: Flask
- **Responsive UI**: Modern, user-friendly interface

## 📊 Dataset

**Dataset**: House Prices: Advanced Regression Techniques
**Source**: [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

Download `train.csv` and place it in the project root before running the notebook.

## 🛠️ Local Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/anilovesdata/HousePrice_Project_MeneAnirejuoritse_23CG034095
cd HousePrice_Project_MeneAnirejuoritse_23CG034095
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
- Download `train.csv` from Kaggle
- Place it in the project root directory (same level as app.py)

5. **Run model training**
```bash
# Navigate to model directory
cd model

python model_building.py
```
Run all cells to train the model and generate required `.pkl` files in the `model/` folder.

6. **Return to project root**
```bash
cd ..
```

7. **Run the Flask app**
```bash
python app.py
```

7. **Access the application**
Open browser and navigate to: `http://localhost:5000`


## 📝 Usage

1. Open the web application
2. Fill in property details:
   - Overall Quality (1-10)
   - Living Area in square feet
   - Basement Area in square feet
   - Garage size (number of cars)
   - Year the house was built
   - Neighborhood
3. Click "Predict House Price"
4. View the estimated sale price


## 📚 Technologies Used

- **Python**: 3.8+
- **Machine Learning**: scikit-learn, Random Forest Regressor
- **Web Framework**: Flask
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib
- **Deployment**: Render/PythonAnywhere/Streamlit Cloud

## 👨‍💻 Author

Mene Anirejuoritse - 23CG034095

## 📄 License

This project is for educational purposes as part of a machine learning course assignment.

## 🙏 Acknowledgments

- Dataset: Kaggle House Prices Competition
- Course: Artificial Intelligence
- Institution: Covenant University