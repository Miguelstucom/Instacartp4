# Instacart Recommendation System

A comprehensive recommendation system for Instacart that combines multiple recommendation approaches to provide personalized product suggestions to users.

## 🚨 Important Notice

The `static` directory containing the trained models and data files is not included in this repository due to GitHub's file size limitations (100MB). You will need to:

1. Download the required data files and place them in the `instacart/static/csv/` directory
2. Train the models using the provided commands
3. The models will be saved in the `instacart/static/model/` directory

## 🛠️ Technologies Used

- **Backend**: Python 3.8+
- **Web Framework**: Django 4.2
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Database**: SQLite (for development)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn, mlxtend
- **Visualization**: Plotly

## 📦 Required Packages


Key packages include:

- django==4.2
- pandas
- numpy
- scikit-learn
- mlxtend
- plotly
- scipy
- joblib

## 📁 Project Structure

```
instacart/
├── static/
│   ├── csv/              # Data files (not included in repo)
│   ├── model/            # Trained models (not included in repo)
│   └── css/              # CSS styles
├── templates/            # HTML templates
├── ml_utils.py          # Machine learning models
├── views.py             # View functions
├── urls.py              # URL routing
└── models.py            # Database models
```

## 🤖 Recommendation Models

The project implements several recommendation approaches:

### 1. Market Basket Analysis (MBA)

- Uses the Apriori algorithm to find frequent itemsets
- Generates association rules based on user purchase patterns
- Considers confidence, lift, and support metrics
- Cluster-specific rules for better personalization

### 2. SVD (Singular Value Decomposition)

- Matrix factorization approach for collaborative filtering
- Handles sparse user-product interaction data
- Includes both global and cluster-specific models
- Uses numpy's SVD implementation for better control

### 3. Cluster-Based Product Recommendations

- Groups users into clusters based on shopping behavior
- Trains separate models for each cluster
- Limits to top 2000 products per cluster for efficiency
- Combines cluster-specific rules with global patterns

## 🚀 Features

- Personalized product recommendations
- Cluster-based user segmentation
- Multiple recommendation approaches
- Interactive web interface
- Performance metrics tracking
- Responsive design

## 💻 Usage

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the models:

```bash
py manage.py train_svd_model --components 40 --test-size 0.2
py manage.py train_model --min-support 0.1 --min-lift 1.5 --test-size 0.2
py manage.py train_svd_cluster_models --components 50 --test-size 0.2
py manage.py train_cluster_product_basket_model --min-support 0.01 --min-lift 1.5 --test-size 0.2
```

4. Run the development server:

```bash
python manage.py runserver
```

## 📊 Model Performance

Each model includes comprehensive metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- RMSE (for SVD)
- MAE (for SVD)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Limitations

- Static directory not included due to size limitations
- Models need to be trained locally
- Requires significant computational resources for training
- Real-time recommendations may have slight latency
