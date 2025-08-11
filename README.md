# 🏠 Mumbai House Price Prediction

A comprehensive machine learning project that predicts house prices in Mumbai's dynamic real estate market using advanced ML algorithms and comprehensive property data analysis.

## 📊 Project Overview

This project analyzes Mumbai's housing market using a dataset of 6,347 property listings with 19 different features. The goal is to build a machine learning model that can accurately predict house prices based on various property characteristics, helping buyers, sellers, and real estate professionals make informed decisions.

## 🎯 Key Features

- **Comprehensive Dataset**: 6,347 properties across 413 unique locations in Mumbai
- **Multiple ML Models**: Linear Regression, Decision Tree, and Random Forest
- **Advanced Preprocessing**: Feature engineering, location encoding, and data transformation
- **Model Deployment**: Saved models for real-time predictions
- **Detailed Analysis**: Complete insights into Mumbai's real estate market

## 📁 Project Structure

```
mumbai house price prediction/
├── README.md                                    # This file
├── Real Estate Price Prediction - Final.ipynb   # Main implementation notebook
├── Real Estate Price Prediction.ipynb           # Alternative approach
├── Mumbai House rent price prediction.ipynb     # Rent price analysis
├── Lab 1 - Project1.ipynb                      # Initial project work
├── House_price.joblib                          # Trained Random Forest model
├── Mumbai1.csv                                 # Main dataset (6,347 properties)
├── Mumbai_House_Rent.csv                       # Rent dataset
├── mumbai_house_price_prediction_blog.md       # Detailed project documentation
└── Outputs of different models/                # Model performance results
```

## 🗃️ Dataset Description

### Core Property Features
- **Price**: Target variable (in Indian Rupees)
- **Area**: Built-up area in square feet
- **Location**: 413 unique locations across Mumbai
- **No. of Bedrooms**: Number of bedrooms (1-4)
- **New/Resale**: Property status indicator

### Amenities (Binary Features)
- Gymnasium
- Lift Available
- Car Parking
- Maintenance Staff
- 24x7 Security
- Children's Play Area
- Clubhouse
- Intercom
- Landscaped Gardens
- Indoor Games
- Gas Connection
- Jogging Track
- Swimming Pool

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Running the Project
1. **Clone or download** the project files
2. **Open** `Real Estate Price Prediction - Final.ipynb` in Jupyter Notebook
3. **Run all cells** to execute the complete analysis
4. **Use the saved model** (`House_price.joblib`) for new predictions

### Making Predictions
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('House_price.joblib')

# Prepare new data (ensure same features as training data)
new_property = pd.DataFrame({
    'Area': [1000],
    'Location': ['Bandra West'],
    'No. of Bedrooms': [2],
    'New/Resale': [1],
    # ... other features
})

# Make prediction
predicted_price = model.predict(new_property)
print(f"Predicted Price: ₹{predicted_price[0]:,.2f}")
```

## 🔬 Machine Learning Approach

### Model Selection
Three different algorithms were tested and compared:

| Model | Mean RMSE | Standard Deviation | Performance |
|-------|-----------|-------------------|-------------|
| Linear Regression | 0.0340 | 0.0007 | Baseline |
| Decision Tree | 0.0259 | 0.0015 | Good |
| **Random Forest** | **0.0196** | **0.0013** | **Best** |

### Key Findings
- **Random Forest Regressor** emerged as the best performing model
- Achieved lowest mean RMSE (0.0196)
- Consistent performance across cross-validation folds
- Good balance between bias and variance

### Why Random Forest Performed Best
- **Non-linear relationships**: Real estate pricing involves complex feature interactions
- **Feature importance**: Can identify which amenities most influence pricing
- **Robustness**: Less prone to overfitting compared to single decision trees

## 📈 Data Analysis Insights

### Price Distribution
- Wide range from affordable to luxury properties
- Right-skewed distribution (typical for real estate markets)
- Most properties in mid-range, fewer high-end luxury properties

### Location Impact
- 413 unique locations capture Mumbai's diverse neighborhoods
- From affordable areas like Kharghar to premium locations
- Location encoding converts categorical data to numerical features

### Amenities Analysis
- Most properties have basic amenities (lift, parking)
- Premium amenities (swimming pool, clubhouse) are less common
- Amenities significantly impact property pricing

## 🛠️ Technical Implementation

### Data Preprocessing
- **Missing Values**: Handled using median imputation
- **Feature Engineering**: Location encoding and price transformation
- **Data Splitting**: 80% training, 20% testing sets

### Pipeline Components
- **Imputation**: Median strategy for missing values
- **Standardization**: Feature scaling to zero mean and unit variance
- **Model Training**: Algorithm fitting with cross-validation

### Model Deployment
- Saved using joblib for future predictions
- Ready for integration into web applications or APIs
- Maintains preprocessing pipeline consistency

## 💼 Business Applications

### For Buyers
- **Price Estimation**: Get fair market value estimates
- **Budget Planning**: Understand price ranges for different locations
- **Negotiation Support**: Data-driven insights for price negotiations

### For Sellers
- **Pricing Strategy**: Set competitive prices based on market data
- **Property Valuation**: Understand feature impact on property value
- **Market Analysis**: Identify pricing trends in neighborhoods

### For Real Estate Professionals
- **Market Intelligence**: Understand pricing factors and trends
- **Client Advisory**: Provide data-backed pricing recommendations
- **Portfolio Management**: Analyze property investments

## 🔮 Future Enhancements

### Additional Features
- **Market Trends**: Time-series data for market dynamics
- **Economic Indicators**: Interest rates, GDP growth, inflation data
- **Infrastructure**: Proximity to metro stations, schools, hospitals
- **Property Images**: Computer vision for condition assessment

### Advanced Models
- **Neural Networks**: Deep learning for complex pattern recognition
- **Ensemble Methods**: Multiple algorithm combinations
- **Time Series Models**: Temporal price variation analysis

### Deployment Options
- **Web Application**: User-friendly prediction interface
- **API Service**: RESTful API for platform integration
- **Mobile App**: On-the-go property valuation tool

## 📚 Files Description

- **`Real Estate Price Prediction - Final.ipynb`**: Complete implementation with best model
- **`House_price.joblib`**: Trained Random Forest model for predictions
- **`Mumbai1.csv`**: Main dataset with 6,347 property listings
- **`mumbai_house_price_prediction_blog.md`**: Detailed project documentation and insights

## 🤝 Contributing

This project is open for contributions! Feel free to:
- Improve model performance
- Add new features or datasets
- Enhance the documentation
- Create web applications or APIs

## 📄 License

This project is for educational and research purposes. The dataset is sourced from Kaggle and should be used in accordance with their terms.

## 👨‍💻 Author

**Shubh Gupta** - Machine Learning enthusiast and data scientist

## 🙏 Acknowledgments

- Dataset source: [Housing Prices in Mumbai](https://www.kaggle.com/datasets/sameep98/housing-prices-in-mumbai)
- Scikit-learn community for excellent ML tools
- Mumbai real estate market insights and analysis

---

**⭐ Star this repository if you find it helpful!**

*This project demonstrates how machine learning can transform real estate analytics, providing valuable insights for various stakeholders in Mumbai's competitive property market.* 