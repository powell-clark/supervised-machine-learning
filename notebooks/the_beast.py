# Core data manipulation and analysis
import numpy as np
import pandas as pd

# Machine Learning - Core
from sklearn.model_selection import (
    train_test_split, 
    KFold, 
)

# Preprocessing and Encoding
from sklearn.preprocessing import (
    OneHotEncoder
)

# Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, 
    r2_score,
)

# Model persistence
import pickle

# Advanced ML models
from xgboost import XGBRegressor

# System utilities
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configure display and plotting options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load the dataframe with outcode information
with open('/home/powell-clark/projects/machine-learning/ML-supervised-learning-showcase/data/df_with_outcode.pkl', 'rb') as f:
    df_with_outcode = pickle.load(f)

@dataclass
class FeatureSet:
    """Container for a feature set configuration"""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    name: str
    description: str

class PreProcessor:
    """Handles initial data transformations and train/test splitting"""
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.outcode_mean_price_per_sqft = pd.Series(index=pd.Index([]), dtype='float64')
        self.global_mean_price_per_sqft = None
    
    def prepare_pre_split_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates features that must be calculated before train/test split:
        - Log transform price
        - Price bands for stratification
        
        Args:
            df: Input DataFrame containing raw features
            
        Returns:
            DataFrame with pre-split features added
        """
        df_processed = df.copy()
        
        # Log transform price
        df_processed['log_price'] = np.log(df_processed['Price'])
        
        # Create price bands for stratification
        df_processed['price_band'] = pd.qcut(df_processed['log_price'], q=10, labels=False)
        
        return df_processed
    
    def create_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs stratified train/test split using price bands.
        
        Args:
            df: DataFrame with price_band feature
            
        Returns:
            Tuple of (train_data, test_data)
        """
        train_data, test_data = train_test_split(
            df,
            test_size=0.2,
            stratify=df['price_band'],
            random_state=self.random_state
        )
        
        return train_data, test_data

class FeatureEncoder:
    """Handles all feature encoding with fold awareness"""
    
    def __init__(self, smoothing_factor: int = 10, min_location_freq: int = 5, random_state: int = RANDOM_STATE):
        self.smoothing_factor = smoothing_factor
        self.min_location_freq = min_location_freq
        self.random_state = random_state
        
    def _encode_house_type(self,
                          fold_train: pd.DataFrame,
                          fold_val: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create one-hot encoding for house type"""
        # Initialize encoder for this fold
        house_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        # Fit on fold's training data
        train_encoded = pd.DataFrame(
            house_encoder.fit_transform(fold_train[['House Type']]),
            columns=house_encoder.get_feature_names_out(['House Type']),
            index=fold_train.index
        )
        
        # Transform validation data
        val_encoded = pd.DataFrame(
            house_encoder.transform(fold_val[['House Type']]),
            columns=house_encoder.get_feature_names_out(['House Type']),
            index=fold_val.index
        )
        
        return {
            'train': train_encoded,
            'val': val_encoded
        }

    def _encode_city_country(self,
                           fold_train: pd.DataFrame,
                           fold_val: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create one-hot encoding for city/county"""
        # Initialize encoder for this fold
        city_country_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        # Fit on fold's training data
        train_encoded = pd.DataFrame(
            city_country_encoder.fit_transform(fold_train[['City/County']]),
            columns=city_country_encoder.get_feature_names_out(['City/County']),
            index=fold_train.index
        )
        
        # Transform validation data
        val_encoded = pd.DataFrame(
            city_country_encoder.transform(fold_val[['City/County']]),
            columns=city_country_encoder.get_feature_names_out(['City/County']),
            index=fold_val.index
        )
        
        return {
            'train': train_encoded,
            'val': val_encoded
        }

    def _encode_outcode_onehot(self,
                              fold_train: pd.DataFrame,
                              fold_val: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create one-hot encoding for outcodes"""
        # Initialize encoder for this fold
        outcode_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        # Fit on fold's training data
        train_encoded = pd.DataFrame(
            outcode_encoder.fit_transform(fold_train[['Outcode']]),
            columns=outcode_encoder.get_feature_names_out(['Outcode']),
            index=fold_train.index
        )
        
        # Transform validation data
        val_encoded = pd.DataFrame(
            outcode_encoder.transform(fold_val[['Outcode']]),
            columns=outcode_encoder.get_feature_names_out(['Outcode']),
            index=fold_val.index
        )
        
        return {
            'train': train_encoded,
            'val': val_encoded
        }

    def _calculate_outcode_price_per_sqft(self,
                                        fold_train: pd.DataFrame,
                                        fold_val: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate mean price per square foot using out-of-fold means for outcodes
        
        Args:
            fold_train: Training data for current fold
            fold_val: Validation data for current fold
            
        Returns:
            Dictionary containing train and validation series of outcode mean price per sqft
        """
        # Initialize empty series for OOF predictions
        oof_price_per_sqft = pd.Series(index=fold_train.index, dtype='float64')
        
        # Calculate OOF means for training data
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in kf.split(fold_train):
            inner_train = fold_train.iloc[train_idx]
            inner_val = fold_train.iloc[val_idx]
            
            # Calculate price per sqft for inner training set
            inner_price_per_sqft = inner_train['Price'] / inner_train['Area in sq ft']
            outcode_means = inner_price_per_sqft.groupby(inner_train['Outcode']).mean()
            global_mean = inner_price_per_sqft.mean()
            
            # Apply to inner validation set
            oof_price_per_sqft.iloc[val_idx] = (
                inner_val['Outcode']
                .map(outcode_means)
                .fillna(global_mean)
            )
        
        # Calculate means for validation data using full training set
        train_price_per_sqft = fold_train['Price'] / fold_train['Area in sq ft']
        outcode_means = train_price_per_sqft.groupby(fold_train['Outcode']).mean()
        global_mean = train_price_per_sqft.mean()
        
        val_price_per_sqft = (
            fold_val['Outcode']
            .map(outcode_means)
            .fillna(global_mean)
        )
        
        return {
            'train': oof_price_per_sqft,
            'val': val_price_per_sqft
        }

    def _encode_geographic(self,
                          fold_train: pd.DataFrame,
                          fold_val: pd.DataFrame) -> Dict[str, Dict[str, pd.Series]]:
        """Create hierarchical encoding for geographic features"""
        # 1. Outcode encoding
        outcode_encoding = self._encode_outcode(fold_train, fold_val)
        
        # 2. Price per square foot feature
        price_per_sqft = self._calculate_outcode_price_per_sqft(fold_train, fold_val)
        
        # 3. Postal code encoding using outcode as prior
        postcode_encoding = self._encode_postcode(
            fold_train, 
            fold_val, 
            outcode_encoding
        )
        
        # 4. Location encoding using postcode as prior
        location_encoding = self._encode_location(
            fold_train,
            fold_val,
            postcode_encoding
        )
        
        return {
            'outcode': outcode_encoding,
            'outcode_price_per_sqft': price_per_sqft,
            'postcode': postcode_encoding,
            'location': location_encoding
        }

    def _encode_outcode(self,
                       fold_train: pd.DataFrame,
                       fold_val: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create target encoding for outcodes using out-of-fold means
        """
        # Fix 1: Specify dtype for empty Series
        oof_predictions = pd.Series(index=fold_train.index, dtype='float64')
        
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in kf.split(fold_train):
            inner_train = fold_train.iloc[train_idx]
            inner_val = fold_train.iloc[val_idx]
            
            outcode_means = inner_train.groupby('Outcode')['log_price'].mean()
            global_mean = inner_train['log_price'].mean()
            
            oof_predictions.iloc[val_idx] = (
                inner_val['Outcode']
                .map(outcode_means)
                .fillna(global_mean)
            )
        
        return {
            'train': oof_predictions,
            'val': fold_val['Outcode'].map(fold_train.groupby('Outcode')['log_price'].mean()).fillna(fold_train['log_price'].mean())
        }
    
    def _encode_postcode(self,
                        fold_train: pd.DataFrame,
                        fold_val: pd.DataFrame,
                        outcode_encoding: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Create hierarchical encoding for postcodes using outcode prior
        """
        # Calculate statistics from training fold
        postcode_means = fold_train.groupby('Postal Code')['log_price'].mean()
        postcode_counts = fold_train['Postal Code'].value_counts()
        
        def encode_postcodes(df: pd.DataFrame, outcode_encoded: pd.Series) -> pd.Series:
            counts = df['Postal Code'].map(postcode_counts)
            means = df['Postal Code'].map(postcode_means)
            
            # Handle unseen categories using outcode encoding
            means = means.fillna(outcode_encoded)
            counts = counts.fillna(0)
            
            # Calculate smoothed values
            weight = counts / (counts + self.smoothing_factor)
            return weight * means + (1 - weight) * outcode_encoded
        
        return {
            'train': encode_postcodes(fold_train, outcode_encoding['train']),
            'val': encode_postcodes(fold_val, outcode_encoding['val'])
        }
    
    def _encode_location(self,
                        fold_train: pd.DataFrame,
                        fold_val: pd.DataFrame,
                        postcode_encoding: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Create hierarchical encoding for locations using postcode prior
        """
        # Calculate statistics from training fold
        location_means = fold_train.groupby('Location')['log_price'].mean()
        location_counts = fold_train['Location'].value_counts()
        
        def encode_locations(df: pd.DataFrame, postcode_encoded: pd.Series) -> pd.Series:
            counts = df['Location'].map(location_counts)
            means = df['Location'].map(location_means)
            
            # Handle missing and unseen locations using postcode encoding
            means = means.fillna(postcode_encoded)
            counts = counts.fillna(0)
            
            # Use postcode encoding for low-frequency locations
            low_freq_mask = (counts < self.min_location_freq) | counts.isna()
            
            # Calculate smoothed values
            weight = counts / (counts + self.smoothing_factor)
            encoded = weight * means + (1 - weight) * postcode_encoded
            
            # Replace low frequency locations with postcode encoding
            encoded[low_freq_mask] = postcode_encoded[low_freq_mask]
            
            return encoded
        
        return {
            'train': encode_locations(fold_train, postcode_encoding['train']),
            'val': encode_locations(fold_val, postcode_encoding['val'])
        }
    
    def create_fold_features(self, fold_train: pd.DataFrame, fold_val: pd.DataFrame) -> List[FeatureSet]:
        """Create all feature set variations for a fold"""
        
        house_features = self._encode_house_type(fold_train, fold_val)
        city_country_features = self._encode_city_country(fold_train, fold_val)
        geo_features = self._encode_geographic(fold_train, fold_val)
        
        feature_combinations = [
            # Base features
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': None,
                'city': None,
                'geo': None,
                'name': 'area_bedrooms',
                'desc': 'Area in sq ft, No. of Bedrooms'
            },
            # Single feature additions
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': house_features,
                'city': None,
                'geo': None,
                'name': 'area_bedrooms_house_onehot',
                'desc': 'Area in sq ft, No. of Bedrooms, House Type (one-hot)'
            },
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': None,
                'city': city_country_features,
                'geo': None,
                'name': 'area_bedrooms_city_onehot',
                'desc': 'Area in sq ft, No. of Bedrooms, City/County (one-hot)'
            },
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': None,
                'city': None,
                'geo': {'outcode': geo_features['outcode']},
                'name': 'area_bedrooms_outcode_target',
                'desc': 'Area in sq ft, No. of Bedrooms, Outcode (target)'
            },
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': None,
                'city': None,
                'geo': {'outcode_onehot': self._encode_outcode_onehot(fold_train, fold_val)},
                'name': 'area_bedrooms_outcode_onehot',
                'desc': 'Area in sq ft, No. of Bedrooms, Outcode (one-hot)'
            },
            # Two feature combinations
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': house_features,
                'city': city_country_features,
                'geo': None,
                'name': 'area_bedrooms_house_city_onehot',
                'desc': 'Area in sq ft, No. of Bedrooms, House Type (one-hot), City/County (one-hot)'
            },
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': house_features,
                'city': None,
                'geo': {'outcode': geo_features['outcode']},
                'name': 'area_bedrooms_house_outcode_target',
                'desc': 'Area in sq ft, No. of Bedrooms, House Type (one-hot), Outcode (target)'
            },
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': house_features,
                'city': None,
                'geo': {'outcode_onehot': self._encode_outcode_onehot(fold_train, fold_val)},
                'name': 'area_bedrooms_house_outcode_onehot',
                'desc': 'Area in sq ft, No. of Bedrooms, House Type (one-hot), Outcode (one-hot)'
            },
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': None,
                'city': city_country_features,
                'geo': {'outcode': geo_features['outcode']},
                'name': 'area_bedrooms_city_outcode_target',
                'desc': 'Area in sq ft, No. of Bedrooms, City/County (one-hot), Outcode (target)'
            },
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': None,
                'city': city_country_features,
                'geo': {'outcode_onehot': self._encode_outcode_onehot(fold_train, fold_val)},
                'name': 'area_bedrooms_city_outcode_onehot',
                'desc': 'Area in sq ft, No. of Bedrooms, City/County (one-hot), Outcode (one-hot)'
            },
            # All features combinations
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': house_features,
                'city': city_country_features,
                'geo': {'outcode': geo_features['outcode']},
                'name': 'area_bedrooms_house_city_outcode_target',
                'desc': 'Area in sq ft, No. of Bedrooms, House Type (one-hot), City/County (one-hot), Outcode (target)'
            },
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': house_features,
                'city': city_country_features,
                'geo': {'outcode_onehot': self._encode_outcode_onehot(fold_train, fold_val)},
                'name': 'area_bedrooms_house_city_outcode_onehot',
                'desc': 'Area in sq ft, No. of Bedrooms, House Type (one-hot), City/County (one-hot), Outcode (one-hot)'
            },
            # Price feature combinations
            {
                'numeric': ['Area in sq ft', 'No. of Bedrooms'],
                'house': house_features,
                'city': city_country_features,
                'geo': {
                    'outcode': geo_features['outcode'],
                    'outcode_price_per_sqft': geo_features['outcode_price_per_sqft']
                },
                'name': 'area_bedrooms_house_city_outcode_target_pricesqft',
                'desc': 'Area in sq ft, No. of Bedrooms, House Type (one-hot), City/County (one-hot), Outcode (target), Price/sqft'
            }
        ]
        
        return [self._combine_features(
            fold_train, 
            fold_val,
            combo['numeric'],
            combo['house'],
            combo['city'],
            combo['geo'],
            combo['name'],
            combo['desc']
        ) for combo in feature_combinations]
    
    def _combine_features(self,
                         fold_train: pd.DataFrame,
                         fold_val: pd.DataFrame,
                         base_numeric: List[str],
                         house_features: Optional[Dict[str, pd.DataFrame]],
                         city_country_features: Optional[Dict[str, pd.DataFrame]],
                         geo_features: Optional[Dict[str, Dict[str, pd.Series]]],
                         name: str,
                         description: str) -> FeatureSet:
        """
        Combine different feature types into a single feature set
        """
        # Start with base numeric features
        X_train = fold_train[base_numeric].copy()
        X_val = fold_val[base_numeric].copy()
        
        # Add house type features if provided
        if house_features:
            X_train = pd.concat([X_train, house_features['train']], axis=1)
            X_val = pd.concat([X_val, house_features['val']], axis=1)

        # Add city/country features if provided
        if city_country_features:
            X_train = pd.concat([X_train, city_country_features['train']], axis=1)
            X_val = pd.concat([X_val, city_country_features['val']], axis=1)
        
        # Add geographic features if provided
        if geo_features:
            for feature_name, feature_dict in geo_features.items():
                # Check if feature is DataFrame (one-hot) or Series (target encoding)
                if isinstance(feature_dict['train'], pd.DataFrame):
                    X_train = pd.concat([X_train, feature_dict['train']], axis=1)
                    X_val = pd.concat([X_val, feature_dict['val']], axis=1)
                else:
                    X_train[feature_name] = feature_dict['train']
                    X_val[feature_name] = feature_dict['val']
        
        return FeatureSet(
            X_train=X_train,
            X_val=X_val,
            y_train=fold_train['log_price'],
            y_val=fold_val['log_price'],
            name=name,
            description=description
        )

class CrossValidator:
    """Handles cross-validation and model evaluation"""
    
    def __init__(self, n_folds: int = 5, random_state: int = RANDOM_STATE):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = {
            'decision_tree': DecisionTreeRegressor(random_state=random_state),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=random_state
            ),
            'xgboost': XGBRegressor(
                n_estimators=100, 
                random_state=random_state
            )
        }
    
    def evaluate_all_combinations(self,
                                train_data: pd.DataFrame,
                                test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate all feature set and model combinations using cross-validation
        """
        kf = KFold(
            n_splits=self.n_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        results = []
        encoder = FeatureEncoder()
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_data)):
            # Split fold data
            fold_train = train_data.iloc[train_idx]
            fold_val = train_data.iloc[val_idx]
            
            # Create feature sets for this fold
            feature_sets = encoder.create_fold_features(fold_train, fold_val)
            
            # Evaluate all models on all feature sets
            fold_results = self._evaluate_fold(
                fold_idx,
                feature_sets
            )
            
            results.extend(fold_results)
        
        return self._create_results_summary(results)

    def _evaluate_fold(self,
                      fold_idx: int,
                      feature_sets: List[FeatureSet]) -> List[Dict]:
        """
        Evaluate all models on all feature sets for a single fold
        """
        results = []
        
        for feature_set in feature_sets:
            for model_name, model in self.models.items():
                # Train model
                model.fit(feature_set.X_train, feature_set.y_train)
                
                # Get predictions
                train_pred_log = model.predict(feature_set.X_train)
                val_pred_log = model.predict(feature_set.X_val)
                
                # Calculate metrics
                train_rmse = self._calculate_rmse(feature_set.y_train, train_pred_log)
                val_rmse = self._calculate_rmse(feature_set.y_val, val_pred_log)
                
                train_r2 = r2_score(feature_set.y_train, train_pred_log)
                val_r2 = r2_score(feature_set.y_val, val_pred_log)
                
                train_pred_price = np.exp(train_pred_log)
                val_pred_price = np.exp(val_pred_log)
                train_true_price = np.exp(feature_set.y_train)
                val_true_price = np.exp(feature_set.y_val)
                
                train_mae = mean_absolute_error(train_true_price, train_pred_price)
                val_mae = mean_absolute_error(val_true_price, val_pred_price)
                
                train_pct_mae = np.mean(np.abs((train_true_price - train_pred_price) / train_true_price)) * 100
                val_pct_mae = np.mean(np.abs((val_true_price - val_pred_price) / val_true_price)) * 100
                
                results.append({
                    'fold': fold_idx,
                    'feature_set': feature_set.name,
                    'description': feature_set.description,
                    'model': model_name,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'train_pct_mae': train_pct_mae,
                    'val_pct_mae': val_pct_mae,
                    'n_features': feature_set.X_train.shape[1]
                })
        
        return results
    
    def _calculate_rmse(self,
                       y_true: pd.Series,
                       y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _create_results_summary(self, results: List[Dict]) -> pd.DataFrame:
        """Create summary DataFrame with statistics across folds"""
        results_df = pd.DataFrame(results)
        
        metrics = ['val_rmse', 'val_r2', 'val_mae', 'val_pct_mae']
        agg_dict = {metric: ['mean', 'std'] for metric in metrics}
        agg_dict['n_features'] = 'first'
        
        summary = results_df.groupby(['feature_set', 'model']).agg(agg_dict).round(4)
        
        # Print detailed results
        print("\nModel Performance Summary:")
        print("==========================")
        for (feature_set, model), group in results_df.groupby(['feature_set', 'model']):
            print(f"\nFeatures: {group['description'].iloc[0]}")
            print(f"Model: {model}")
            print(f"Number of features: {int(group['n_features'].iloc[0])}")
            print(f"R² Score: {float(group['val_r2'].mean()):.3f} (±{float(group['val_r2'].std()):.3f})")
            print(f"RMSE (log price): {float(group['val_rmse'].mean()):.3f} (±{float(group['val_rmse'].std()):.3f})")
            print(f"MAE (£): {float(group['val_mae'].mean()):,.0f} (±{float(group['val_mae'].std()):,.0f})")
            print(f"Percentage Error: {float(group['val_pct_mae'].mean()):.1f}% (±{float(group['val_pct_mae'].std()):.1f}%)")
            print("-" * 80)
        
        return summary

def run_model_comparison_pipeline(df_with_outcode: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Run complete pipeline from raw data to model comparison"""
    
    preprocessor = PreProcessor()
    validator = CrossValidator()
    
    # Create pre-split features
    df_processed = preprocessor.prepare_pre_split_features(df_with_outcode)
    
    # Create initial train/test split
    train_data, test_data = preprocessor.create_train_test_split(df_processed)
    
    # Run cross-validation evaluation
    results = validator.evaluate_all_combinations(train_data, test_data)
    
    # Store preprocessing artifacts
    artifacts = {
        'outcode_mean_price_per_sqft': preprocessor.outcode_mean_price_per_sqft,
        'global_mean_price_per_sqft': preprocessor.global_mean_price_per_sqft,
        'train_shape': train_data.shape,
        'test_shape': test_data.shape
    }
    
    return results, artifacts

# Run pipeline
results, artifacts = run_model_comparison_pipeline(df_with_outcode)

# Display results
print("\nModel Performance Summary:")
for (feature_set, model), group in results.groupby(['feature_set', 'model']):
    print(f"\n{feature_set} - {model}:")
    print(f"R² Score: {float(group['val_r2']['mean']):.3f} (±{float(group['val_r2']['std']):.3f})")
    print(f"RMSE (log price): {float(group['val_rmse']['mean']):.3f} (±{float(group['val_rmse']['std']):.3f})")
    print(f"MAE (£): {float(group['val_mae']['mean']):,.0f} (±{float(group['val_mae']['std']):,.0f})")
    print(f"Percentage Error: {float(group['val_pct_mae']['mean']):.1f}% (±{float(group['val_pct_mae']['std']):.1f}%)")

# Display artifact information
print("\nPreprocessing Artifacts:")
print(f"Outcode Mean Price per Sqft: {artifacts['outcode_mean_price_per_sqft']}")
print(f"Global Mean Price per Sqft: {artifacts['global_mean_price_per_sqft']}")
print(f"Training Set Shape: {artifacts['train_shape']}")
print(f"Test Set Shape: {artifacts['test_shape']}")



# This completes the implementation. The pipeline:
# 1. Handles all preprocessing with proper train/test splitting
# 2. Creates all feature variations within each cross-validation fold
# 3. Prevents data leakage in encodings
# 4. Provides comprehensive evaluation metrics

# Would you like me to:
# 1. Add error handling and validation checks?
# 2. Add visualization functions for the results?
# 3. Add functionality for final model training after feature selection?
