import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

np.random.seed(42)

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df

def clean_data(df, remove_missing=True):
    df = df.drop(columns=['Unnamed: 0'])
    df['Location'].fillna('Unknown', inplace=True)
    
    # Extract postcode outcode
    df['Postcode Outcode'] = df['Postal Code'].str.split().str[0]
    
    numeric_cols = ['Price', 'Area in sq ft', 'No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if remove_missing:
        df.dropna(subset=numeric_cols, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    
    print(f"After cleaning: {len(df)} rows remaining")
    return df

def explore_data(df):
    print("\nStatistical Summary:")
    print(df.describe())
    
    plt.figure(figsize=(10,6))
    sns.histplot(df['Price'], kde=True)
    plt.title('Distribution of House Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.scatter(df['Area in sq ft'], df['Price'], alpha=0.5)
    plt.title('Price vs. Area')
    plt.xlabel('Area in sq ft')
    plt.ylabel('Price')
    plt.show()
    
    # Normal correlation matrix (numeric variables only)
    numeric_cols = ['Price', 'Area in sq ft', 'No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions']
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix (Numeric Variables)')
    plt.show()

    print("\nCorrelation Matrix (Numeric Variables):")
    print(corr_matrix)

    # Enhanced correlation matrix (including encoded categorical variables)
    categorical_cols = ['House Type', 'Location', 'City/County', 'Postcode Outcode']
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in categorical_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    enhanced_corr_columns = numeric_cols + categorical_cols
    enhanced_corr_matrix = df_encoded[enhanced_corr_columns].corr()

    plt.figure(figsize=(12,10))
    sns.heatmap(enhanced_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Enhanced Correlation Matrix (Including Encoded Categorical Variables)')
    plt.show()

    print("\nEnhanced Correlation Matrix (Including Encoded Categorical Variables):")
    print(enhanced_corr_matrix)

    # Analyze categorical variables
    for col in categorical_cols:
        print(f"\nTop 5 {col} by Average Price:")
        print(df.groupby(col)['Price'].mean().sort_values(ascending=False).head())

    # Additional visualizations for categorical variables
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=col, y='Price', data=df.sort_values('Price', ascending=False).head(100))
        plt.xticks(rotation=90)
        plt.title(f'Price Distribution by {col} (Top 100 Expensive Properties)')
        plt.show()

def prepare_data(df):
    features = ['Area in sq ft', 'No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions', 'House Type', 'Location', 'City/County', 'Postcode Outcode']
    target = 'Price'

    X = df[features].copy()  # Create a copy to avoid SettingWithCopyWarning
    y = df[target]

    le = LabelEncoder()
    for col in ['House Type', 'Location', 'City/County', 'Postcode Outcode']:
        X[col] = le.fit_transform(X[col].astype(str))

    # Analyze the impact of Postcode Outcode
    outcode_price_avg = df.groupby('Postcode Outcode')['Price'].mean().sort_values(ascending=False)
    print("\nTop 5 Postcode Outcodes by Average Price:")
    print(outcode_price_avg.head())
    print("\nBottom 5 Postcode Outcodes by Average Price:")
    print(outcode_price_avg.tail())

    # Visualize the impact of Postcode Outcode
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Postcode Outcode', y='Price', data=df.sort_values('Price', ascending=False).head(100))
    plt.xticks(rotation=90)
    plt.title('Price Distribution by Postcode Outcode (Top 100 Expensive Properties)')
    plt.savefig('postcode_outcode_impact.png', bbox_inches='tight')
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type='decision_tree'):
    if model_type == 'decision_tree':
        model = DecisionTreeRegressor(random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_type.capitalize()} Model Evaluation:")
    print(f"Root Mean Squared Error: £{rmse:.2f}")
    print(f"R-squared Score: {r2:.4f}")

    plt.figure(figsize=(10,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Predicted vs Actual House Prices ({model_type.capitalize()})")
    plt.show()

    if model_type == 'random_forest':
        feature_importance = model.feature_importances_
        feature_names = X_train.columns
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10,6))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('Feature Importance (Random Forest)')
        plt.show()

    return rmse, r2

def main():
    df = load_data('data/London.csv')

    # Approach 1: Remove missing values and use Decision Tree
    df_clean = clean_data(df, remove_missing=True)
    explore_data(df_clean)
    X_train, X_test, y_train, y_test = prepare_data(df_clean)
    rmse_dt, r2_dt = train_and_evaluate_model(X_train, X_test, y_train, y_test, 'decision_tree')

    # Approach 2: Keep all rows, impute missing values, and use Random Forest
    df_all = clean_data(df, remove_missing=False)
    X = df_all[['Area in sq ft', 'No. of Bedrooms', 'No. of Bathrooms', 'No. of Receptions', 'House Type', 'Location', 'City/County', 'Postcode Outcode']].copy()  # Create a copy to avoid SettingWithCopyWarning
    y = df_all['Price']

    le = LabelEncoder()
    for col in ['House Type', 'Location', 'City/County', 'Postcode Outcode']:
        X[col] = le.fit_transform(X[col].astype(str))

    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    rmse_rf, r2_rf = train_and_evaluate_model(X_train, X_test, y_train, y_test, 'random_forest')

    print("\nComparison of approaches:")
    print(f"Decision Tree (removing missing values): RMSE = £{rmse_dt:.2f}, R2 = {r2_dt:.4f}")
    print(f"Random Forest (keeping all rows): RMSE = £{rmse_rf:.2f}, R2 = {r2_rf:.4f}")

    print("\nAnalysis complete. All plots have been displayed in separate windows.")

if __name__ == '__main__':
    main()