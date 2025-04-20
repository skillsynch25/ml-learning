import streamlit as st
st.set_page_config(page_title="ML Learning Platform", layout="wide")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from models import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    train_lightgbm
)
from deep_learning_models import (
    CNNModel,
    RNNModel,
    NLPModel,
    Autoencoder,
    GAN
)
from utils import (
    preprocess_data,
    plot_correlation_matrix,
    plot_feature_distributions,
    plot_prediction_vs_actual,
    plot_residuals,
    generate_model_summary
)
from deep_learning_visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_feature_maps,
    plot_word_cloud,
    plot_attention_weights,
    plot_embeddings,
    plot_gan_samples,
    plot_autoencoder_reconstruction,
    plot_model_architecture
)

# Try to import CatBoost, but don't fail if it's not available
try:
    from models import train_catboost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    st.warning("CatBoost is not available. Some features may be limited.")

import torch
import torch.nn as nn
import math
from nlp_models import TextPreprocessor, SentimentAnalyzer, WordFrequencyAnalyzer, TextClassifier, TextGenerator

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, bidirectional):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,  # Explicitly specify num_layers parameter
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
    
    def forward(self, x):
        try:
            # Ensure input is a tensor and on the correct device
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            
            # Get device from input tensor
            device = x.device
            
            # Initialize hidden state and cell state
            batch_size = x.size(0)
            h0 = torch.zeros(
                self.num_layers * 2 if self.bidirectional else self.num_layers,
                batch_size,
                self.hidden_dim,
                device=device
            )
            
            c0 = torch.zeros(
                self.num_layers * 2 if self.bidirectional else self.num_layers,
                batch_size,
                self.hidden_dim,
                device=device
            )
            
            # Forward propagate LSTM
            # LSTM returns (output, (hidden_state, cell_state))
            lstm_out, _ = self.lstm(x, (h0, c0))  # Unpack tuple: output, (hn, cn)
            
            # Take the last time step's output
            last_output = lstm_out[:, -1, :]  # Shape: [batch_size, hidden_dim * num_directions]
            
            # Pass through linear layer
            out = self.fc(last_output)  # Shape: [batch_size, output_dim]
            
            return out
            
        except Exception as e:
            st.error(f"Error in LSTM forward pass: {str(e)}")
            st.write(f"Error occurred with input shape: {x.shape if hasattr(x, 'shape') else 'No shape'}")
            st.write(f"Error occurred with input type: {type(x)}")
            raise

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation):
        super().__init__()
        
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layer
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # Input embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Take the last time step's output
        x = x[:, -1, :]
        
        # Final prediction
        return self.fc(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Create positional encodings on demand
        self.register_buffer('pe', None)
        
    def _create_pe(self, max_len):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(max_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe
        
    def forward(self, x):
        # Get sequence length
        seq_len = x.size(0)
        
        # Create or update positional encodings if needed
        if self.pe is None or self.pe.size(0) < seq_len:
            self.pe = self._create_pe(seq_len)
            
        # Add positional encoding
        x = x + self.pe[:seq_len]
        return self.dropout(x)

def main():
    st.title("Machine Learning Learning Platform")
    
    # Sidebar for model selection and parameters
    st.sidebar.header("Model Configuration")
    
    # Model type selection
    model_category = st.sidebar.selectbox(
        "Select Model Category",
        ["Traditional ML", "Deep Learning", "NLP"]
    )
    
    # Initialize parameters with default values
    params = {
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.001,
        'optimizer': "Adam",
        'validation_split': 0.2,
        'early_stopping': True,
        'patience': 5,
        'data_augmentation': False,
        'use_tensorboard': True,
        'save_best_model': True,
        'model_save_path': "best_model.h5",
        # RNN-specific parameters
        'num_rnn_layers': 2,
        'hidden_units': 128,
        'dropout_rate': 0.2,
        'bidirectional': True,
        'sequence_length': 20,
        'return_sequences': False,
        'stateful': False,
        'lstm_sequence_length': 10,
        'lstm_units': 128,
        'lstm_dropout': 0.2,
        'lstm_bidirectional': True
    }
    
    # Deep Learning Model Selection (if Deep Learning category is selected)
    if model_category == "Deep Learning":
        dl_model_type = st.sidebar.selectbox(
            "Select Deep Learning Model",
            [
                "CNN (Convolutional Neural Network)",
                "RNN (Recurrent Neural Network)",
                "LSTM (Long Short-Term Memory)",
                "GRU (Gated Recurrent Unit)",
                "Transformer",
                "Autoencoder",
                "GAN (Generative Adversarial Network)",
                "ResNet",
                "DenseNet",
                "MobileNet"
            ]
        )
        
        # Common Parameters
        st.sidebar.subheader("Common Parameters")
        params['batch_size'] = st.sidebar.slider(
            "Batch Size",
            min_value=16,
            max_value=256,
            value=32,
            step=16
        )
        params['epochs'] = st.sidebar.slider(
            "Number of Epochs",
            min_value=1,
            max_value=100,
            value=10,
            step=1
        )
        params['learning_rate'] = st.sidebar.slider(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            format="%.4f"
        )
        params['optimizer'] = st.sidebar.selectbox(
            "Optimizer",
            ["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta"]
        )
        
        # Model-Specific Parameters
        st.sidebar.subheader("Model-Specific Parameters")
        
        if "RNN" in dl_model_type:
            st.sidebar.write("#### RNN Parameters")
            params['num_rnn_layers'] = st.sidebar.slider(
                "Number of RNN Layers",
                min_value=1,
                max_value=4,
                value=2
            )
            params['hidden_units'] = st.sidebar.slider(
                "Hidden Units",
                min_value=32,
                max_value=512,
                value=128,
                step=32
            )
            params['dropout_rate'] = st.sidebar.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.1
            )
            params['bidirectional'] = st.sidebar.checkbox("Use Bidirectional Layers", value=True)
            
            # Additional RNN-specific parameters
            st.sidebar.write("#### Additional Parameters")
            params['sequence_length'] = st.sidebar.slider(
                "Sequence Length",
                min_value=5,
                max_value=100,
                value=20
            )
            params['return_sequences'] = st.sidebar.checkbox("Return Sequences", value=False)
            params['stateful'] = st.sidebar.checkbox("Stateful RNN", value=False)
            
        elif "LSTM" in dl_model_type:
            st.sidebar.write("#### LSTM Parameters")
            params['num_lstm_layers'] = st.sidebar.slider(
                "Number of LSTM Layers",
                min_value=1,
                max_value=4,
                value=2
            )
            params['lstm_units'] = st.sidebar.slider(
                "LSTM Units",
                min_value=32,
                max_value=512,
                value=128,
                step=32
            )
            params['lstm_dropout'] = st.sidebar.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.1
            )
            params['lstm_recurrent_dropout'] = st.sidebar.slider(
                "Recurrent Dropout Rate",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.1
            )
            params['lstm_bidirectional'] = st.sidebar.checkbox("Use Bidirectional LSTM", value=True)
            
            # Additional LSTM-specific parameters
            st.sidebar.write("#### Additional Parameters")
            params['lstm_sequence_length'] = st.sidebar.slider(
                "Sequence Length",
                min_value=5,
                max_value=100,
                value=20
            )
            params['lstm_return_sequences'] = st.sidebar.checkbox("Return Sequences", value=False)
            params['lstm_stateful'] = st.sidebar.checkbox("Stateful LSTM", value=False)
            params['lstm_implementation'] = st.sidebar.selectbox(
                "LSTM Implementation",
                ["1", "2"],
                help="Implementation mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas implementation 2 will batch them into fewer, larger operations."
            )
        
        # Training Parameters
        st.sidebar.subheader("Training Parameters")
        params['validation_split'] = st.sidebar.slider(
            "Validation Split",
            min_value=0.1,
            max_value=0.3,
            value=0.2,
            step=0.05
        )
        params['early_stopping'] = st.sidebar.checkbox("Use Early Stopping", value=True)
        if params['early_stopping']:
            params['patience'] = st.sidebar.slider(
                "Early Stopping Patience",
                min_value=3,
                max_value=10,
                value=5
            )
        params['data_augmentation'] = st.sidebar.checkbox("Use Data Augmentation", value=False)
        
        # Callback Options
        st.sidebar.subheader("Callbacks")
        params['use_tensorboard'] = st.sidebar.checkbox("Use TensorBoard", value=True)
        params['save_best_model'] = st.sidebar.checkbox("Save Best Model", value=True)
        if params['save_best_model']:
            params['model_save_path'] = st.sidebar.text_input(
                "Model Save Path",
                value="best_model.h5"
            )
    
    # NLP Model Selection (if NLP category is selected)
    if model_category == "NLP":
        nlp_model_type = st.sidebar.selectbox(
            "Select NLP Model",
            [
                "BERT (Base)",
                "BERT (Large)",
                "GPT-2",
                "RoBERTa",
                "DistilBERT",
                "T5",
                "BART",
                "XLNet"
            ]
        )
        
        # NLP Processing Parameters
        st.sidebar.subheader("NLP Processing Parameters")
        
        # Text Cleaning Options
        st.sidebar.write("#### Text Cleaning")
        params['remove_numbers'] = st.sidebar.checkbox("Remove Numbers", value=True)
        params['remove_special_chars'] = st.sidebar.checkbox("Remove Special Characters", value=True)
        params['lowercase'] = st.sidebar.checkbox("Convert to Lowercase", value=True)
        
        # Tokenization Options
        st.sidebar.write("#### Tokenization")
        params['tokenization_method'] = st.sidebar.selectbox(
            "Tokenization Method",
            ["Word", "Sentence", "Both"]
        )
        
        # Stopword Options
        st.sidebar.write("#### Stopwords")
        params['remove_stopwords'] = st.sidebar.checkbox("Remove Stopwords", value=True)
        params['custom_stopwords'] = st.sidebar.text_area(
            "Custom Stopwords (one per line)",
            value=""
        )
        
        # Word Normalization
        st.sidebar.write("#### Word Normalization")
        params['normalization_method'] = st.sidebar.selectbox(
            "Normalization Method",
            ["Lemmatization", "Stemming"]
        )
        
        # Sentiment Analysis Options
        st.sidebar.write("#### Sentiment Analysis")
        params['sentiment_chunk_size'] = st.sidebar.slider(
            "Chunk Size",
            min_value=100,
            max_value=1000,
            value=500,
            step=100
        )
        params['max_sentences'] = st.sidebar.slider(
            "Max Sentences to Analyze",
            min_value=10,
            max_value=100,
            value=50
        )
        
        # NER Options
        st.sidebar.write("#### Named Entity Recognition")
        params['ner_chunk_size'] = st.sidebar.slider(
            "NER Chunk Size",
            min_value=100,
            max_value=1000,
            value=500,
            step=100
        )
        params['show_entity_visualization'] = st.sidebar.checkbox(
            "Show Entity Visualization",
            value=True
        )
        
        # Word Cloud Options
        st.sidebar.write("#### Word Cloud")
        params['max_words'] = st.sidebar.slider(
            "Max Words",
            min_value=50,
            max_value=500,
            value=200
        )
        params['wordcloud_width'] = st.sidebar.slider(
            "Width",
            min_value=400,
            max_value=1200,
            value=800,
            step=100
        )
        params['wordcloud_height'] = st.sidebar.slider(
            "Height",
            min_value=200,
            max_value=800,
            value=400,
            step=100
        )
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "txt", "png", "jpg"])
    
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            data = pd.read_csv(uploaded_file)
            st.sidebar.success("Dataset uploaded successfully!")
            
            # Display dataset info
            st.subheader("Dataset Overview")
            st.write(f"Number of rows: {data.shape[0]}")
            st.write(f"Number of columns: {data.shape[1]}")
            
            # Data type information
            st.write("#### Data Types")
            data_types = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes,
                'Unique Values': data.nunique(),
                'Missing Values': data.isnull().sum()
            })
            st.dataframe(data_types)
            
            # Separate numerical and categorical columns
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = data.select_dtypes(include=['object', 'category', 'string']).columns
            
            # Display first few rows
            st.write("First few rows:")
            st.dataframe(data.head())
            
            # Data visualization
            st.subheader("Data Visualization")
            
            # Numerical data visualization
            if len(numerical_cols) > 0:
                st.write("##### Numerical Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Correlation matrix
                    if len(numerical_cols) >= 2:
                        try:
                            # Select only numerical columns for correlation
                            numeric_data = data[numerical_cols]
                            st.pyplot(plot_correlation_matrix(numeric_data))
                        except Exception as e:
                            st.warning(f"Could not create correlation matrix: {str(e)}")
                    else:
                        st.warning("Need at least 2 numerical columns for correlation matrix")
                
                with col2:
                    # Feature distributions
                    st.plotly_chart(plot_feature_distributions(data[numerical_cols]))
            else:
                st.warning("No numerical columns found for correlation analysis")
            
            if model_category == "Traditional ML":
                handle_traditional_ml(data)
            elif model_category == "Deep Learning":
                handle_deep_learning(data, dl_model_type, params)
            elif model_category == "NLP":
                handle_nlp(data)
        elif uploaded_file.type == "text/plain":
            # Handle text file for NLP
            text_content = uploaded_file.read().decode("utf-8")
            st.sidebar.success("Text file uploaded successfully!")
            
            # Display text content
            st.subheader("Text Content")
            st.text_area("Uploaded Text", text_content, height=200)
            
            if model_category == "NLP":
                handle_nlp_text(text_content, nlp_model_type, {
                    'remove_numbers': params['remove_numbers'],
                    'remove_special_chars': params['remove_special_chars'],
                    'lowercase': params['lowercase'],
                    'tokenization_method': params['tokenization_method'],
                    'remove_stopwords': params['remove_stopwords'],
                    'custom_stopwords': params['custom_stopwords'],
                    'normalization_method': params['normalization_method'],
                    'sentiment_chunk_size': params['sentiment_chunk_size'],
                    'max_sentences': params['max_sentences'],
                    'ner_chunk_size': params['ner_chunk_size'],
                    'show_entity_visualization': params['show_entity_visualization'],
                    'max_words': params['max_words'],
                    'wordcloud_width': params['wordcloud_width'],
                    'wordcloud_height': params['wordcloud_height']
                })
            else:
                st.warning("Please select 'NLP' as the model category for text analysis.")
        else:
            # Handle image data
            st.image(uploaded_file, caption="Uploaded Image")
            if model_category == "Deep Learning":
                handle_image_deep_learning(uploaded_file)

def handle_traditional_ml(data):
    # Feature selection
    st.subheader("Feature Selection")
    target_column = st.selectbox("Select target column", data.columns)
    feature_columns = st.multiselect("Select feature columns", 
                                   [col for col in data.columns if col != target_column],
                                   default=[col for col in data.columns if col != target_column])
    
    if target_column and feature_columns:
        # Data preprocessing
        X = data[feature_columns]
        y = data[target_column]
        
        # Preprocess data with detailed analysis
        X_scaled, y, scaler = preprocess_data(X, y)
        
        # Train-test split parameters
        st.sidebar.subheader("Training Configuration")
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, key="test_size")
        
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        
        # Model selection in sidebar
        st.sidebar.header("Model Configuration")
        model_types = ["Linear Regression", "Random Forest", "XGBoost", "LightGBM"]
        if CATBOOST_AVAILABLE:
            model_types.append("CatBoost")
        
        model_type = st.sidebar.selectbox(
            "Select Model Type",
            model_types,
            key="model_type_select"
        )
        
        # Model parameters in sidebar
        st.sidebar.subheader("Model Parameters")
        
        if model_type == "Linear Regression":
            st.sidebar.write("#### Linear Regression Parameters")
            fit_intercept = st.sidebar.checkbox("Fit Intercept", value=True, key="lr_fit_intercept")
            normalize = st.sidebar.checkbox("Normalize", value=False, key="lr_normalize")
            
        elif model_type == "Random Forest":
            st.sidebar.write("#### Random Forest Parameters")
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, key="rf_n_estimators")
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 10, key="rf_max_depth")
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2, key="rf_min_samples_split")
            min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 1, key="rf_min_samples_leaf")
            
        elif model_type == "XGBoost":
            st.sidebar.write("#### XGBoost Parameters")
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, key="xgb_learning_rate")
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, key="xgb_n_estimators")
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 6, key="xgb_max_depth")
            subsample = st.sidebar.slider("Subsample", 0.5, 1.0, 1.0, key="xgb_subsample")
            colsample_bytree = st.sidebar.slider("Colsample by Tree", 0.5, 1.0, 1.0, key="xgb_colsample")
            
        elif model_type == "LightGBM":
            st.sidebar.write("#### LightGBM Parameters")
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, key="lgbm_learning_rate")
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, key="lgbm_n_estimators")
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 6, key="lgbm_max_depth")
            num_leaves = st.sidebar.slider("Number of Leaves", 20, 100, 31, key="lgbm_num_leaves")
            subsample = st.sidebar.slider("Subsample", 0.5, 1.0, 1.0, key="lgbm_subsample")
            
        elif model_type == "CatBoost" and CATBOOST_AVAILABLE:
            st.sidebar.write("#### CatBoost Parameters")
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, key="catboost_learning_rate")
            iterations = st.sidebar.slider("Number of Trees", 10, 200, 100, key="catboost_iterations")
            depth = st.sidebar.slider("Max Depth", 1, 20, 6, key="catboost_depth")
            l2_leaf_reg = st.sidebar.slider("L2 Regularization", 0.0, 10.0, 3.0, key="catboost_l2_leaf_reg")
            border_count = st.sidebar.slider("Border Count", 32, 255, 128, key="catboost_border_count")
        
        # Train model
        with st.spinner("Training model..."):
            if model_type == "Linear Regression":
                model = train_linear_regression(X_train, y_train, fit_intercept, normalize)
            elif model_type == "Random Forest":
                model = train_random_forest(X_train, y_train, n_estimators, max_depth, min_samples_split, min_samples_leaf)
            elif model_type == "XGBoost":
                model = train_xgboost(X_train, y_train, learning_rate, n_estimators, max_depth, subsample, colsample_bytree)
            elif model_type == "LightGBM":
                model = train_lightgbm(X_train, y_train, learning_rate, n_estimators, max_depth, num_leaves, subsample)
            elif model_type == "CatBoost" and CATBOOST_AVAILABLE:
                model = train_catboost(X_train, y_train, learning_rate, iterations, depth, l2_leaf_reg, border_count)
            
            # Model evaluation
            st.subheader("Model Evaluation")
            y_pred = model.predict(X_test)
            
            # Model summary
            st.subheader("Model Performance Summary")
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            ev_score = explained_variance_score(y_test, y_pred)
            
            metrics = {
                "Mean Squared Error (MSE)": f"{mse:.4f}",
                "Root Mean Squared Error (RMSE)": f"{rmse:.4f}",
                "R² Score": f"{r2:.4f}",
                "Mean Absolute Error (MAE)": f"{mae:.4f}",
                "Explained Variance Score": f"{ev_score:.4f}"
            }
            
            # Create a nice metrics display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{r2:.4f}")
            with col2:
                st.metric("RMSE", f"{rmse:.4f}")
            with col3:
                st.metric("MAE", f"{mae:.4f}")
            
            # Detailed metrics explanation
            st.subheader("Metrics Explanation")
            st.write("""
            - **R² Score**: Indicates how well the model fits the data (1.0 is perfect fit)
            - **RMSE**: Average prediction error in the same unit as the target variable
            - **MAE**: Average absolute prediction error
            - **Explained Variance**: How much of the target variance is captured by the model
            """)
            
            # Prediction Analysis
            st.subheader("Prediction Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Actual vs Predicted
                fig_actual_pred = px.scatter(
                    x=y_test, y=y_pred,
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    title='Actual vs Predicted Values'
                )
                fig_actual_pred.add_shape(
                    type='line', line=dict(dash='dash'),
                    x0=min(y_test), y0=min(y_test),
                    x1=max(y_test), y1=max(y_test)
                )
                st.plotly_chart(fig_actual_pred)
            
            with col2:
                # Residuals Plot
                residuals = y_test - y_pred
                fig_residuals = px.scatter(
                    x=y_pred, y=residuals,
                    labels={'x': 'Predicted Values', 'y': 'Residuals'},
                    title='Residuals Analysis'
                )
                fig_residuals.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig_residuals)
            
            # Residuals Distribution
            st.subheader("Residuals Distribution")
            fig_residuals_dist = px.histogram(
                residuals, 
                title='Distribution of Residuals',
                labels={'value': 'Residuals', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_residuals_dist)
            
            # SHAP Analysis
            st.subheader("SHAP Feature Importance Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                explainer = shap.Explainer(model)
                shap_values = explainer(X_test)
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_test, feature_names=feature_columns, show=False)
                st.pyplot(fig)
                plt.clf()
            
            with col2:
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_test, feature_names=feature_columns, plot_type="bar", show=False)
                st.pyplot(fig)
                plt.clf()
            
            # LIME Analysis
            st.subheader("LIME Feature Importance Analysis")
            
            # Create LIME explainer
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_scaled,
                feature_names=feature_columns,
                class_names=[target_column],
                mode='regression'
            )
            
            # Explain multiple instances
            st.write("LIME Explanations for Sample Predictions:")
            num_examples = min(5, len(X_test))
            
            for i in range(num_examples):
                exp = lime_explainer.explain_instance(X_test[i], model.predict)
                
                st.write(f"### Sample {i+1}")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show actual vs predicted values
                    st.write(f"Actual Value: {y_test.iloc[i]:.4f}")
                    st.write(f"Predicted Value: {y_pred[i]:.4f}")
                    st.write(f"Difference: {abs(y_test.iloc[i] - y_pred[i]):.4f}")
                
                with col2:
                    # Plot LIME explanation
                    fig = exp.as_pyplot_figure()
                    st.pyplot(fig)
                    plt.clf()
            
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importances = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importances,
                    x='Feature',
                    y='Importance',
                    title='Feature Importance Scores'
                )
                st.plotly_chart(fig)
            
            # Cross-Validation Analysis
            st.subheader("Cross-Validation Analysis")
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            st.write("5-Fold Cross-Validation Results:")
            cv_results = pd.DataFrame({
                'Fold': range(1, 6),
                'Score': cv_scores
            })
            
            fig = px.bar(
                cv_results,
                x='Fold',
                y='Score',
                title='Cross-Validation Scores by Fold'
            )
            st.plotly_chart(fig)
            
            st.write(f"Mean CV Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
            
            # Learning Curves
            st.subheader("Learning Curves")
            from sklearn.model_selection import learning_curve
            
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5, n_jobs=-1
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_mean,
                name='Training Score',
                mode='lines+markers'
            ))
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=test_mean,
                name='Cross-Validation Score',
                mode='lines+markers'
            ))
            fig.update_layout(
                title='Learning Curves',
                xaxis_title='Training Examples',
                yaxis_title='Score'
            )
            st.plotly_chart(fig)
            
            # Model Insights Summary
            st.subheader("Model Insights Summary")
            st.write("""
            ### Key Findings:
            1. **Model Performance**
               - Overall R² score indicates how well the model explains the variance in the target variable
               - RMSE and MAE show the average prediction error in actual units
            
            2. **Feature Importance**
               - SHAP values show how each feature contributes to individual predictions
               - The feature importance plot shows which variables are most influential overall
            
            3. **Prediction Reliability**
               - The residuals plot helps identify any systematic prediction errors
               - The cross-validation scores show how well the model generalizes
            
            4. **Model Behavior**
               - LIME explanations provide local interpretability for individual predictions
               - Learning curves show how the model's performance scales with data size
            
            ### Recommendations:
            """)
            
            # Generate recommendations based on the analysis
            if r2 < 0.5:
                st.write("- Consider feature engineering or collecting additional relevant features")
                st.write("- Try more complex model architectures")
            
            if np.abs(residuals).max() > 2 * rmse:
                st.write("- Investigate outliers in the predictions")
                st.write("- Consider robust regression techniques")
            
            if cv_scores.std() > 0.1:
                st.write("- The model's performance varies significantly across folds")
                st.write("- Consider collecting more data or using regularization")
            
            # Export options
            st.subheader("Export Results")
            
            # Create a summary DataFrame
            summary_data = pd.DataFrame({
                'Metric': metrics.keys(),
                'Value': metrics.values()
            })
            
            # Convert to CSV
            csv = summary_data.to_csv(index=False)
            st.download_button(
                label="Download Metrics Summary",
                data=csv,
                file_name="model_metrics_summary.csv",
                mime="text/csv"
            )

def handle_deep_learning(data, dl_model_type, params):
    st.subheader("Deep Learning Model Configuration")
    
    # Data Preprocessing
    st.write("### Data Preprocessing")
    
    # Select target column with unique key
    target_column = st.selectbox("Select target column", data.columns, key="dl_target_column")
    
    # Select feature columns with unique key
    feature_columns = st.multiselect(
        "Select feature columns",
        [col for col in data.columns if col != target_column],
        default=[col for col in data.columns if col != target_column],
        key="dl_feature_columns"
    )
    
    if target_column and feature_columns:
        # Data type handling
        st.write("#### Data Type Handling")
        categorical_columns = []
        numerical_columns = []
        
        for col in feature_columns:
            if data[col].dtype in ['object', 'category', 'string']:
                categorical_columns.append(col)
            else:
                numerical_columns.append(col)
        
        # Create a copy of the data for preprocessing
        X = data[feature_columns].copy()
        y = data[target_column]
        
        if categorical_columns:
            st.write("Categorical columns detected:", categorical_columns)
            encoding_method = st.selectbox(
                "Select encoding method for categorical variables",
                ["One-Hot Encoding", "Label Encoding", "Ordinal Encoding"],
                key="dl_categorical_encoding"
            )
            
            # Encode categorical variables
            if encoding_method == "One-Hot Encoding":
                X = pd.get_dummies(X, columns=categorical_columns)
            elif encoding_method == "Label Encoding":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for col in categorical_columns:
                    X[col] = le.fit_transform(X[col].astype(str))
            elif encoding_method == "Ordinal Encoding":
                from sklearn.preprocessing import OrdinalEncoder
                oe = OrdinalEncoder()
                X[categorical_columns] = oe.fit_transform(X[categorical_columns].astype(str))
        
        # Scale numerical features
        if numerical_columns:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
        
        # Create sequences
        sequence_length = params['sequence_length']
        def create_sequences(data, sequence_length):
            sequences = []
            for i in range(len(data) - sequence_length):
                sequences.append(data[i:(i + sequence_length)])
            return np.array(sequences)
        
        # Convert input data to numpy array first
        X_array = np.array(X.values, dtype=np.float32)
        y_array = np.array(y.values, dtype=np.float32)  # Convert target to float32 for regression
        
        # Create sequences
        X_sequences = create_sequences(X_array, sequence_length)
        y_sequences = y_array[sequence_length:]
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_sequences, y_sequences, test_size=params['validation_split'], random_state=42
        )
        
        # Ensure proper shape for RNN input
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Convert to PyTorch tensors
        import torch
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train).view(-1, 1)
        y_test = torch.FloatTensor(y_test).view(-1, 1)
        
        # Model selection and handling
        if "RNN" in dl_model_type:
            handle_rnn(data, params, X_train, X_test, y_train, y_test)
        elif "LSTM" in dl_model_type:
            handle_lstm(data, params, X_train, X_test, y_train, y_test)
        elif "CNN" in dl_model_type:
            handle_cnn(data, params, X_train, X_test, y_train, y_test)
        elif "GRU" in dl_model_type:
            handle_gru(data, params, X_train, X_test, y_train, y_test)
        elif "Transformer" in dl_model_type:
            handle_transformer(data, params, X_train, X_test, y_train, y_test)
        elif "Autoencoder" in dl_model_type:
            handle_autoencoder(data, params, X_train, X_test, y_train, y_test)
        elif "GAN" in dl_model_type:
            handle_gan(data, params, X_train, X_test, y_train, y_test)
        else:
            st.error(f"Unsupported model type: {dl_model_type}")

def handle_cnn(data, params, X_train, X_test, y_train, y_test):
    st.subheader("CNN Model Configuration")
    
    # Feature and target selection
    st.write("Select input features and target variable:")
    target_column = st.selectbox("Select target column", data.columns)
    feature_columns = st.multiselect("Select feature columns", 
                                   [col for col in data.columns if col != target_column],
                                   default=[col for col in data.columns if col != target_column])
    
    if target_column and feature_columns:
        # Prepare data
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Model parameters
        input_shape = st.sidebar.text_input("Input Shape (e.g., 28,28,1)", "28,28,1")
        num_classes = st.sidebar.number_input("Number of Classes", min_value=2, value=10)
        
        # Create and train model
        if st.sidebar.button("Train CNN"):
            input_shape = tuple(map(int, input_shape.split(',')))
            cnn_model = CNNModel(input_shape, num_classes)
            
            # Training parameters
            epochs = st.sidebar.slider("Epochs", 1, 50, 10)
            batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
            
            # Train model
            with st.spinner("Training CNN..."):
                try:
                    # Reshape data if needed
                    if len(X.shape) != 4:  # CNN expects (samples, height, width, channels)
                        try:
                            X = X.reshape(-1, *input_shape)
                        except ValueError:
                            st.error(f"Could not reshape input data to shape {(-1,) + input_shape}. Please check your input shape and data dimensions.")
                            return
                    
                    history = cnn_model.train(X, y, epochs=epochs, batch_size=batch_size)
                    
                    # Visualizations
                    st.subheader("Training Results")
                    st.pyplot(plot_training_history(history))
                    
                    # Feature maps
                    st.subheader("Feature Maps")
                    st.pyplot(plot_feature_maps(cnn_model.model, "conv2d", X[:1]))
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    st.write("Please check your data format and model parameters.")

def handle_rnn(data, params, X_train, X_test, y_train, y_test):
    st.subheader("RNN Model Configuration")
    
    # Model Architecture Visualization
    st.write("### Model Architecture")
    st.write("""
    The RNN model consists of three main components:
    
    1. **Input Layer**: Takes sequences of features
       - Input shape: (batch_size, sequence_length, input_dim)
       - Each time step processes one feature vector
    
    2. **RNN Layer**: Processes sequences with hidden states
       - Hidden state size: {hidden_dim}
       - Number of layers: {num_layers}
       - Bidirectional: {bidirectional}
       - Dropout rate: {dropout}
    
    3. **Dense Layer**: Produces final predictions
       - Output dimension: 1 (for regression)
    
    The RNN cell processes information using the following equations:
    - Hidden state: h(t) = tanh(W_hh * h(t-1) + W_xh * x(t) + b_h)
    - Output: y(t) = W_hy * h(t) + b_y
    
    Where:
    - W_hh: Hidden-to-hidden weights
    - W_xh: Input-to-hidden weights
    - W_hy: Hidden-to-output weights
    - b_h, b_y: Bias terms
    - h(t-1): Previous hidden state
    - x(t): Current input
    - h(t): Current hidden state
    - y(t): Current output
    """.format(
        hidden_dim=params['hidden_units'],
        num_layers=params['num_rnn_layers'],
        bidirectional=params['bidirectional'],
        dropout=params['dropout_rate']
    ))
    
    # Interactive RNN Cell Visualization
    st.write("### RNN Cell Visualization")
    st.write("""
    The RNN cell processes information through time using the following mechanism:
    
    1. **Input Processing**:
       - Each time step receives an input vector x(t)
       - The input is combined with the previous hidden state h(t-1)
       - A non-linear transformation (tanh) is applied
    
    2. **Hidden State Update**:
       - The hidden state h(t) captures information from previous time steps
       - It serves as the memory of the network
       - The update is controlled by learnable weights W_hh and W_xh
    
    3. **Output Generation**:
       - The hidden state h(t) is transformed into the output y(t)
       - For sequence prediction, we use the last time step's output
       - The transformation is controlled by weights W_hy
    """)
    
    # Initialize model with correct parameters
    input_dim = X_train.shape[2]
    hidden_dim = params['hidden_units']
    num_layers = params['num_rnn_layers']
    output_dim = 1
    dropout = params['dropout_rate']
    bidirectional = params['bidirectional']
    
    # Create RNN model
    model = RNNModel(input_dim, hidden_dim, num_layers, output_dim, dropout, bidirectional)
    
    # Training configuration
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training process
    num_epochs = params['epochs']
    batch_size = params['batch_size']
    
    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Training loop
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            train_losses = []
            val_losses = []
            learning_rates = []
            batch_losses = []
            
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0
                batch_loss_history = []
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_loss_history.append(loss.item())
                
                # Store metrics
                train_losses.append(epoch_loss / len(train_loader))
                batch_losses.extend(batch_loss_history)
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test)
                    val_loss = criterion(val_outputs, y_test)
                    val_losses.append(val_loss.item())
                
                # Update learning rate
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)
                
                # Display progress
                st.write(f"Epoch {epoch+1}/{num_epochs}")
                st.write(f"Training Loss: {train_losses[-1]:.4f}")
                st.write(f"Validation Loss: {val_losses[-1]:.4f}")
                st.write(f"Learning Rate: {current_lr:.6f}")
                
                # Plot training progress
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Plot loss curves
                ax1.plot(train_losses, label='Training Loss')
                ax1.plot(val_losses, label='Validation Loss')
                ax1.set_title('Loss Curves')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                
                # Plot learning rate
                ax2.plot(learning_rates)
                ax2.set_title('Learning Rate Schedule')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                
                st.pyplot(fig)
                plt.close()
                
                # Early stopping check
                if len(val_losses) > 10 and val_losses[-1] > min(val_losses[:-10]):
                    st.write("Early stopping triggered")
                    break
            
            # Final model evaluation
            st.write("### Model Evaluation")
            model.eval()
            with torch.no_grad():
                predictions = model(X_test)
                mse = criterion(predictions, y_test)
                rmse = torch.sqrt(mse)
                
                # Calculate additional metrics
                mae = torch.mean(torch.abs(predictions - y_test))
                r2 = 1 - mse / torch.var(y_test)
                
                st.write(f"Final MSE: {mse.item():.4f}")
                st.write(f"Final RMSE: {rmse.item():.4f}")
                st.write(f"Final MAE: {mae.item():.4f}")
                st.write(f"R² Score: {r2.item():.4f}")
            
            # Plot predictions vs actual
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test.numpy(), predictions.numpy(), alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted Values')
            st.pyplot(fig)
            plt.close()
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates
            }, 'rnn_model.pth')
            
            st.success("Training completed and model saved!")

def handle_lstm(data, params, X_train, X_test, y_train, y_test):
    st.subheader("LSTM Model Configuration")
    
    # Model Architecture Visualization
    st.write("### Model Architecture")
    st.write("""
    The LSTM model consists of three main components:
    
    1. **Input Layer**: Takes sequences of features
       - Input shape: (batch_size, sequence_length, input_dim)
       - Each time step processes one feature vector
    
    2. **LSTM Layer**: Processes sequences with memory cells
       - Hidden state size: {hidden_dim}
       - Number of layers: {num_layers}
       - Bidirectional: {bidirectional}
       - Dropout rate: {dropout}
    
    3. **Dense Layer**: Produces final predictions
       - Output dimension: 1 (for regression)
    
    The LSTM cell processes information using the following equations:
    - Forget Gate: f(t) = σ(W_f * [h(t-1), x(t)] + b_f)
    - Input Gate: i(t) = σ(W_i * [h(t-1), x(t)] + b_i)
    - Cell State: C̃(t) = tanh(W_C * [h(t-1), x(t)] + b_C)
    - Cell Update: C(t) = f(t) * C(t-1) + i(t) * C̃(t)
    - Output Gate: o(t) = σ(W_o * [h(t-1), x(t)] + b_o)
    - Hidden State: h(t) = o(t) * tanh(C(t))
    
    Where:
    - σ: Sigmoid activation
    - W_f, W_i, W_C, W_o: Weight matrices
    - b_f, b_i, b_C, b_o: Bias terms
    """.format(
        hidden_dim=params['hidden_units'],
        num_layers=params['num_lstm_layers'],
        bidirectional=params['bidirectional'],
        dropout=params['dropout_rate']
    ))
    
    # Interactive LSTM Cell Visualization
    st.write("### LSTM Cell Visualization")
    st.write("""
    The LSTM cell processes information through time using the following mechanism:
    
    1. **Gate Operations**:
       - Forget Gate: Decides what information to discard
       - Input Gate: Decides what new information to store
       - Output Gate: Decides what information to output
    
    2. **Memory Cell**:
       - Maintains long-term dependencies
       - Updates based on gate activations
       - Preserves information across time steps
    
    3. **Information Flow**:
       - Input → Gates → Memory Cell → Output
       - Selective information processing
       - Adaptive memory management
    """)
    
    # Initialize model with correct parameters
    input_dim = X_train.shape[2]
    hidden_dim = params['hidden_units']
    num_layers = params['num_lstm_layers']
    output_dim = 1
    dropout = params['dropout_rate']
    bidirectional = params['bidirectional']
    
    # Create LSTM model
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout, bidirectional)
    
    # Training configuration
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training process
    num_epochs = params['epochs']
    batch_size = params['batch_size']
    
    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Training loop
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            train_losses = []
            val_losses = []
            learning_rates = []
            batch_losses = []
            
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0
                batch_loss_history = []
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_loss_history.append(loss.item())
                
                # Store metrics
                train_losses.append(epoch_loss / len(train_loader))
                batch_losses.extend(batch_loss_history)
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test)
                    val_loss = criterion(val_outputs, y_test)
                    val_losses.append(val_loss.item())
                
                # Update learning rate
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)
                
                # Display progress
                st.write(f"Epoch {epoch+1}/{num_epochs}")
                st.write(f"Training Loss: {train_losses[-1]:.4f}")
                st.write(f"Validation Loss: {val_losses[-1]:.4f}")
                st.write(f"Learning Rate: {current_lr:.6f}")
                
                # Plot training progress
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Plot loss curves
                ax1.plot(train_losses, label='Training Loss')
                ax1.plot(val_losses, label='Validation Loss')
                ax1.set_title('Loss Curves')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                
                # Plot learning rate
                ax2.plot(learning_rates)
                ax2.set_title('Learning Rate Schedule')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                
                st.pyplot(fig)
                plt.close()
                
                # Early stopping check
                if len(val_losses) > 10 and val_losses[-1] > min(val_losses[:-10]):
                    st.write("Early stopping triggered")
                    break
            
            # Final model evaluation
            st.write("### Model Evaluation")
            model.eval()
            with torch.no_grad():
                predictions = model(X_test)
                mse = criterion(predictions, y_test)
                rmse = torch.sqrt(mse)
                
                # Calculate additional metrics
                mae = torch.mean(torch.abs(predictions - y_test))
                r2 = 1 - mse / torch.var(y_test)
                
                st.write(f"Final MSE: {mse.item():.4f}")
                st.write(f"Final RMSE: {rmse.item():.4f}")
                st.write(f"Final MAE: {mae.item():.4f}")
                st.write(f"R² Score: {r2.item():.4f}")
            
            # Plot predictions vs actual
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test.numpy(), predictions.numpy(), alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted Values')
            st.pyplot(fig)
            plt.close()
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates
            }, 'lstm_model.pth')
            
            st.success("Training completed and model saved!")

def handle_gru(data, params):
    st.subheader("GRU Model Configuration")
    
    # Model parameters
    input_dim = st.sidebar.number_input("Input Dimension", min_value=1, value=10)
    hidden_dim = st.sidebar.number_input("Hidden Dimension", min_value=1, value=64)
    num_layers = st.sidebar.number_input("Number of Layers", min_value=1, value=2)
    num_classes = st.sidebar.number_input("Number of Classes", min_value=2, value=10)
    
    # Create and train model
    if st.sidebar.button("Train GRU"):
        gru_model = GRUModel(input_dim, hidden_dim, num_layers, num_classes)
        
        # Training parameters
        epochs = st.sidebar.slider("Epochs", 1, 50, 10)
        batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
        
        # Train model
        with st.spinner("Training GRU..."):
            gru_model.train(data, epochs=epochs, batch_size=batch_size)
            
            # Visualizations
            st.subheader("Model Architecture")
            st.image(plot_model_architecture(gru_model.model))

def handle_transformer(data, params, X_train, X_test, y_train, y_test):
    st.subheader("Transformer Model Implementation")
    
    # 1. Model Architecture Explanation
    st.write("### 1. Transformer Architecture Overview")
    st.write("""
    The Transformer model consists of several key components:
    
    1. **Input Embedding Layer**: Converts input sequences into dense vectors
    2. **Positional Encoding**: Adds positional information to the embeddings
    3. **Encoder Layers**: Each containing:
       - Multi-Head Self-Attention
       - Feed-Forward Network
       - Layer Normalization
       - Residual Connections
    4. **Decoder Layers**: Similar to encoder but with:
       - Masked Multi-Head Self-Attention
       - Encoder-Decoder Attention
    5. **Output Layer**: Produces final predictions
    
    For our regression task, we'll use a simplified version focusing on the encoder part.
    """)
    
    # 2. Model Parameters Configuration
    st.write("### 2. Model Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        params['d_model'] = st.number_input(
            "Model Dimension (d_model)",
            min_value=32,
            max_value=512,
            value=128,
            step=32,
            help="Dimension of the model's hidden states"
        )
        params['nhead'] = st.number_input(
            "Number of Attention Heads",
            min_value=1,
            max_value=16,
            value=4,
            help="Number of parallel attention heads"
        )
        params['num_encoder_layers'] = st.number_input(
            "Number of Encoder Layers",
            min_value=1,
            max_value=6,
            value=2,
            help="Number of transformer encoder layers"
        )
    
    with col2:
        params['dim_feedforward'] = st.number_input(
            "Feed-Forward Dimension",
            min_value=64,
            max_value=2048,
            value=512,
            step=64,
            help="Dimension of the feed-forward network"
        )
        params['dropout'] = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="Dropout probability"
        )
        params['activation'] = st.selectbox(
            "Activation Function",
            ["relu", "gelu"],
            help="Activation function in the feed-forward network"
        )
    
    # 3. Model Implementation
    st.write("### 3. Model Implementation")
    st.write("""
    ```python
    class TransformerModel(nn.Module):
        def __init__(self, input_dim, d_model, nhead, num_encoder_layers, 
                     dim_feedforward, dropout, activation):
            super().__init__()
            
            # Input embedding layer
            self.embedding = nn.Linear(input_dim, d_model)
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, dropout)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers
            )
            
            # Output layer
            self.fc = nn.Linear(d_model, 1)
            
        def forward(self, x):
            # Input embedding
            x = self.embedding(x)
            
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Transformer encoder
            x = self.transformer_encoder(x)
            
            # Take the last time step's output
            x = x[:, -1, :]
            
            # Final prediction
            return self.fc(x)
    ```
    """)
    
    # 4. Positional Encoding Explanation
    st.write("### 4. Positional Encoding")
    st.write("""
    Positional encoding adds information about the position of each token in the sequence:
    
    ```python
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            self.d_model = d_model
            
            # Create positional encodings on demand
            self.register_buffer('pe', None)
            
        def _create_pe(self, max_len):
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
            pe = torch.zeros(max_len, 1, self.d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            return pe
            
        def forward(self, x):
            # Get sequence length
            seq_len = x.size(0)
            
            # Create or update positional encodings if needed
            if self.pe is None or self.pe.size(0) < seq_len:
                self.pe = self._create_pe(seq_len)
                
            # Add positional encoding
            x = x + self.pe[:seq_len]
            return self.dropout(x)
    ```
    
    This creates a unique pattern for each position using sine and cosine functions of different frequencies.
    """)
    
    # 5. Training Process
    st.write("### 5. Training Process")
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            # Initialize model
            model = TransformerModel(
                input_dim=X_train.shape[2],
                d_model=params['d_model'],
                nhead=params['nhead'],
                num_encoder_layers=params['num_encoder_layers'],
                dim_feedforward=params['dim_feedforward'],
                dropout=params['dropout'],
                activation=params['activation']
            )
            
            # Training configuration
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=5, factor=0.5
            )
            
            # Training metrics
            train_losses = []
            val_losses = []
            learning_rates = []
            
            # Training loop
            for epoch in range(params['epochs']):
                model.train()
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test)
                    val_loss = criterion(val_outputs, y_test)
                
                # Update learning rate
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Store metrics
                train_losses.append(loss.item())
                val_losses.append(val_loss.item())
                learning_rates.append(current_lr)
                
                if (epoch + 1) % 5 == 0:
                    st.write(f"Epoch [{epoch+1}/{params['epochs']}], "
                           f"Train Loss: {loss.item():.4f}, "
                           f"Val Loss: {val_loss.item():.4f}, "
                           f"LR: {current_lr:.6f}")
            
            # 6. Training Analysis
            st.write("### 6. Training Analysis")
            
            # Loss curves
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ax1.plot(train_losses, label='Training Loss')
            ax1.plot(val_losses, label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Learning rate schedule
            ax2.plot(learning_rates)
            ax2.set_title('Learning Rate Schedule')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            st.pyplot(fig)
            
            # 7. Model Evaluation
            st.write("### 7. Model Evaluation")
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                
                # Calculate metrics
                mse = criterion(y_pred, y_test)
                rmse = torch.sqrt(mse)
                mae = torch.mean(torch.abs(y_pred - y_test))
                
                # R² Score
                ss_res = torch.sum((y_test - y_pred) ** 2)
                ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
                r2 = 1 - ss_res / ss_tot
                
                # Explained Variance
                explained_variance = 1 - torch.var(y_test - y_pred) / torch.var(y_test)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MSE", f"{mse.item():.4f}")
                    st.metric("RMSE", f"{rmse.item():.4f}")
                with col2:
                    st.metric("MAE", f"{mae.item():.4f}")
                    st.metric("R² Score", f"{r2.item():.4f}")
                with col3:
                    st.metric("Explained Variance", f"{explained_variance.item():.4f}")
            
            # 8. Prediction Analysis
            st.write("### 8. Prediction Analysis")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Actual vs Predicted
            ax1.scatter(y_test.numpy(), y_pred.numpy(), alpha=0.5)
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title('Actual vs Predicted Values')
            
            # Residuals Plot
            residuals = y_test - y_pred
            ax2.scatter(y_pred.numpy(), residuals.numpy(), alpha=0.5)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals Analysis')
            st.pyplot(fig)
            
            # 9. Attention Visualization
            st.write("### 9. Attention Visualization")
            try:
                # Get attention weights from the first encoder layer
                attention_weights = model.transformer_encoder.layers[0].self_attn.attn_weights
                
                # Plot attention matrix for a sample sequence
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(attention_weights[0].mean(dim=0).detach().numpy(),
                             cmap='viridis')
                ax.set_title('Attention Weights (First Head)')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                plt.colorbar(im)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not visualize attention weights: {str(e)}")
            
            # 10. Model Interpretation
            st.write("### 10. Model Interpretation")
            st.write("""
            #### Key Insights:
            
            1. **Architecture Design**:
               - The model uses self-attention to capture dependencies between different time steps
               - Positional encoding helps maintain sequence order information
               - Multiple encoder layers allow for hierarchical feature learning
            
            2. **Training Dynamics**:
               - The loss curves show how well the model is learning
               - Learning rate scheduling helps fine-tune the model
            
            3. **Performance Metrics**:
               - R² score indicates how well the model explains the variance
               - RMSE and MAE provide different perspectives on prediction errors
            
            4. **Attention Patterns**:
               - The attention visualization shows how the model focuses on different parts of the sequence
               - This helps understand which time steps are most relevant for predictions
            
            5. **Residual Analysis**:
               - The residuals plot helps identify systematic prediction errors
               - This can guide improvements in model architecture or training
            """)
            
            # 11. Model Saving
            st.write("### 11. Model Saving")
            if st.button("Save Model"):
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'params': params
                }, 'transformer_model.pth')
                st.success("Model saved successfully!")

def handle_autoencoder(data, params):
    st.subheader("Autoencoder Configuration")
    
    # Model parameters
    input_dim = st.sidebar.number_input("Input Dimension", min_value=1, value=784)
    encoding_dim = st.sidebar.number_input("Encoding Dimension", min_value=1, value=32)
    
    # Create and train model
    if st.sidebar.button("Train Autoencoder"):
        autoencoder = Autoencoder(input_dim, encoding_dim)
        
        # Training parameters
        epochs = st.sidebar.slider("Epochs", 1, 50, 10)
        batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
        
        # Train model
        with st.spinner("Training Autoencoder..."):
            history = autoencoder.train(data, epochs=epochs, batch_size=batch_size)
            
            # Visualizations
            st.subheader("Training Results")
            st.pyplot(plot_training_history(history))
            
            # Reconstruction
            st.subheader("Reconstruction Results")
            reconstructed = autoencoder.autoencoder.predict(data[:5])
            st.pyplot(plot_autoencoder_reconstruction(data[:5], reconstructed))

def handle_gan(data, params):
    st.subheader("GAN Configuration")
    
    # Model parameters
    latent_dim = st.sidebar.number_input("Latent Dimension", min_value=1, value=100)
    img_shape = st.sidebar.text_input("Image Shape (e.g., 28,28,1)", "28,28,1")
    
    # Create and train model
    if st.sidebar.button("Train GAN"):
        img_shape = tuple(map(int, img_shape.split(',')))
        gan = GAN(latent_dim, img_shape)
        
        # Training parameters
        epochs = st.sidebar.slider("Epochs", 1, 100, 50)
        batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
        
        # Train model
        with st.spinner("Training GAN..."):
            generator, discriminator = gan.train(data, epochs=epochs, batch_size=batch_size)
            
            # Visualizations
            st.subheader("Generated Samples")
            st.pyplot(plot_gan_samples(generator, latent_dim))

def handle_resnet(data, params):
    st.subheader("ResNet Model Configuration")
    
    # Model parameters
    num_blocks = st.sidebar.slider(
        "Number of Residual Blocks",
        min_value=2,
        max_value=8,
        value=4
    )
    block_filters = st.sidebar.text_input(
        "Filters per Block (comma-separated)",
        value="64,128,256,512"
    )
    use_bottleneck = st.sidebar.checkbox("Use Bottleneck Blocks", value=True)
    
    # Create and train model
    if st.sidebar.button("Train ResNet"):
        resnet_model = ResNetModel(num_blocks, [int(x) for x in block_filters.split(',')], use_bottleneck)
        
        # Training parameters
        epochs = st.sidebar.slider("Epochs", 1, 50, 10)
        batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
        
        # Train model
        with st.spinner("Training ResNet..."):
            resnet_model.train(data, epochs=epochs, batch_size=batch_size)
            
            # Visualizations
            st.subheader("Model Architecture")
            st.image(plot_model_architecture(resnet_model.model))

def handle_densenet(data, params):
    st.subheader("DenseNet Model Configuration")
    
    # Model parameters
    num_blocks = st.sidebar.slider(
        "Number of Dense Blocks",
        min_value=2,
        max_value=8,
        value=4
    )
    block_filters = st.sidebar.text_input(
        "Filters per Block (comma-separated)",
        value="64,128,256,512"
    )
    growth_rate = st.sidebar.slider(
        "Growth Rate",
        min_value=16,
        max_value=64,
        value=32
    )
    use_bottleneck = st.sidebar.checkbox("Use Bottleneck Blocks", value=True)
    
    # Create and train model
    if st.sidebar.button("Train DenseNet"):
        densenet_model = DenseNetModel(num_blocks, [int(x) for x in block_filters.split(',')], growth_rate, use_bottleneck)
        
        # Training parameters
        epochs = st.sidebar.slider("Epochs", 1, 50, 10)
        batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
        
        # Train model
        with st.spinner("Training DenseNet..."):
            densenet_model.train(data, epochs=epochs, batch_size=batch_size)
            
            # Visualizations
            st.subheader("Model Architecture")
            st.image(plot_model_architecture(densenet_model.model))

def handle_mobilenet(data, params):
    st.subheader("MobileNet Model Configuration")
    
    # Model parameters
    alpha = st.sidebar.slider(
        "Width Multiplier",
        min_value=0.25,
        max_value=1.0,
        value=1.0,
        step=0.25
    )
    depth_multiplier = st.sidebar.slider(
        "Depth Multiplier",
        min_value=1,
        max_value=2,
        value=1
    )
    include_top = st.sidebar.checkbox("Include Top Layers", value=True)
    
    # Create and train model
    if st.sidebar.button("Train MobileNet"):
        mobilenet_model = MobileNetModel(alpha, depth_multiplier, include_top)
        
        # Training parameters
        epochs = st.sidebar.slider("Epochs", 1, 50, 10)
        batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
        
        # Train model
        with st.spinner("Training MobileNet..."):
            mobilenet_model.train(data, epochs=epochs, batch_size=batch_size)
            
            # Visualizations
            st.subheader("Model Architecture")
            st.image(plot_model_architecture(mobilenet_model.model))

def handle_image_deep_learning(image_file):
    st.subheader("Image Processing")
    
    # Model selection
    image_model_type = st.selectbox(
        "Select Image Model",
        ["CNN", "Autoencoder", "GAN"]
    )
    
    try:
        # Convert BytesIO to numpy array
        import io
        from PIL import Image
        import numpy as np
        
        # Read the image
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        if image_model_type == "CNN":
            handle_cnn_image(image_array)
        elif image_model_type == "Autoencoder":
            handle_autoencoder_image(image_array)
        elif image_model_type == "GAN":
            handle_gan_image(image_array)
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.write("Please make sure you've uploaded a valid image file (PNG, JPG, or JPEG).")

def handle_cnn_image(image_array):
    st.subheader("CNN Image Processing")
    
    # Preprocess the image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image
    
    # Resize image to expected size
    img = image.array_to_img(image_array)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Load pre-trained model
    from tensorflow.keras.applications import ResNet50
    model = ResNet50(weights='imagenet')
    
    # Make prediction
    preds = model.predict(img_array)
    from tensorflow.keras.applications.resnet50 import decode_predictions
    
    # Display top 5 predictions
    st.subheader("Top 5 Predictions")
    decoded_preds = decode_predictions(preds, top=5)[0]
    
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        st.write(f"{i+1}. {label}: {score:.2f}")

def handle_autoencoder_image(image_array):
    st.subheader("Autoencoder Image Processing")
    st.info("Autoencoder image processing is not implemented yet. Coming soon!")

def handle_gan_image(image_array):
    st.subheader("GAN Image Processing")
    st.info("GAN image processing is not implemented yet. Coming soon!")

def handle_nlp_text(text_content, nlp_model_type, params):
    st.subheader("NLP Analysis")
    
    # Model Information
    st.write("### Step 1: Model Information")
    from transformers import (
        BertTokenizer, BertModel,
        GPT2Tokenizer, GPT2Model,
        RobertaTokenizer, RobertaModel,
        DistilBertTokenizer, DistilBertModel,
        T5Tokenizer, T5Model,
        BartTokenizer, BartModel,
        XLNetTokenizer, XLNetModel
    )
    
    @st.cache_resource
    def load_model_and_tokenizer(model_type):
        if model_type == "BERT (Base)":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
        elif model_type == "BERT (Large)":
            tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            model = BertModel.from_pretrained('bert-large-uncased')
        elif model_type == "GPT-2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2Model.from_pretrained('gpt2')
        elif model_type == "RoBERTa":
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model = RobertaModel.from_pretrained('roberta-base')
        elif model_type == "DistilBERT":
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        elif model_type == "T5":
            tokenizer = T5Tokenizer.from_pretrained('t5-base')
            model = T5Model.from_pretrained('t5-base')
        elif model_type == "BART":
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
            model = BartModel.from_pretrained('facebook/bart-base')
        elif model_type == "XLNet":
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            model = XLNetModel.from_pretrained('xlnet-base-cased')
        return tokenizer, model
    
    with st.spinner(f"Loading {nlp_model_type}..."):
        tokenizer, model = load_model_and_tokenizer(nlp_model_type)
        st.success(f"{nlp_model_type} loaded successfully!")
    
    # Model Information Display
    col1, col2 = st.columns(2)
    with col1:
        st.write("Model Type:", nlp_model_type)
        st.write("Tokenizer:", tokenizer.__class__.__name__)
    with col2:
        st.write("Model Architecture:", model.__class__.__name__)
        st.write("Parameters:", f"{model.num_parameters():,}")
    
    # Text Preprocessing Steps
    st.write("### Step 2: Text Preprocessing")
    
    # 1.1 Basic Text Cleaning
    st.write("#### 1.1 Basic Text Cleaning")
    import re
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.stem import PorterStemmer
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    # Clean text
    def clean_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    cleaned_text = clean_text(text_content)
    st.write("Text after basic cleaning:")
    st.text_area("Cleaned Text", cleaned_text[:1000] + "...", height=200)
    
    # 1.2 Tokenization
    st.write("#### 1.2 Tokenization")
    words = word_tokenize(cleaned_text)
    sentences = sent_tokenize(cleaned_text)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("First 10 words:")
        st.write(words[:10])
    with col2:
        st.write("First 3 sentences:")
        st.write(sentences[:3])
    
    # 1.3 Stopword Removal
    st.write("#### 1.3 Stopword Removal")
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    st.write("Words before stopword removal:", len(words))
    st.write("Words after stopword removal:", len(filtered_words))
    
    # 1.4 Lemmatization
    st.write("#### 1.4 Lemmatization")
    lemmatizer = WordNetLemmatizer()
    
    # Download required NLTK data
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    
    # Lemmatize words with proper POS tagging
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                   "N": nltk.corpus.wordnet.NOUN,
                   "V": nltk.corpus.wordnet.VERB,
                   "R": nltk.corpus.wordnet.ADV}
        return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)
    
    with st.spinner("Lemmatizing words..."):
        lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in filtered_words]
    
    # 1.5 Stemming
    st.write("#### 1.5 Stemming")
    stemmer = PorterStemmer()
    with st.spinner("Stemming words..."):
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    # Compare original, lemmatized, and stemmed words
    comparison_df = pd.DataFrame({
        'Original': filtered_words[:20],
        'Lemmatized': lemmatized_words[:20],
        'Stemmed': stemmed_words[:20]
    })
    st.dataframe(comparison_df)
    
    # Show POS tags for sample words
    st.write("#### Part-of-Speech Tags")
    sample_words = filtered_words[:10]
    pos_tags = nltk.pos_tag(sample_words)
    pos_df = pd.DataFrame(pos_tags, columns=['Word', 'POS Tag'])
    st.dataframe(pos_df)
    
    # Show differences between lemmatization and stemming
    st.write("#### Lemmatization vs Stemming Comparison")
    differences = []
    for orig, lem, stem in zip(filtered_words[:20], lemmatized_words[:20], stemmed_words[:20]):
        if lem != stem:
            differences.append({
                'Word': orig,
                'Lemmatized': lem,
                'Stemmed': stem
            })
    if differences:
        diff_df = pd.DataFrame(differences)
        st.dataframe(diff_df)
    else:
        st.info("No differences found between lemmatization and stemming in the sample words.")
    
    # 1.6 Word Frequency Analysis
    st.write("#### 1.6 Word Frequency Analysis")
    from collections import Counter
    word_freq = Counter(lemmatized_words)
    
    # Display top 20 most common words
    st.write("Top 20 Most Common Words:")
    top_words = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])
    fig = px.bar(top_words, x='Word', y='Frequency', title='Word Frequency Distribution')
    st.plotly_chart(fig)
    
    # 1.7 Word Cloud
    st.write("#### 1.7 Word Cloud Visualization")
    from wordcloud import WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    # 1.8 Text Statistics
    st.write("#### 1.8 Text Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Characters", len(text_content))
    with col2:
        st.metric("Number of Words", len(words))
    with col3:
        st.metric("Number of Sentences", len(sentences))
    
    # 1.9 Vocabulary Analysis
    st.write("#### 1.9 Vocabulary Analysis")
    unique_words = set(lemmatized_words)
    st.write(f"Unique words: {len(unique_words)}")
    st.write(f"Vocabulary density (unique words / total words): {len(unique_words)/len(words):.2%}")
    
    # 1.10 Sentence Length Analysis
    st.write("#### 1.10 Sentence Length Analysis")
    sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
    fig = px.histogram(
        x=sentence_lengths,
        title='Distribution of Sentence Lengths',
        labels={'x': 'Words per Sentence', 'y': 'Count'}
    )
    st.plotly_chart(fig)
    
    # Sentiment Analysis
    st.write("### Step 3: Sentiment Analysis")
    from transformers import pipeline
    
    with st.spinner("Loading sentiment analyzer..."):
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Process text in smaller chunks for sentiment analysis
    chunk_size = 500  # Process 500 characters at a time
    sentiments = []
    
    with st.spinner("Analyzing sentiment..."):
        for i in range(0, len(text_content), chunk_size):
            chunk = text_content[i:i + chunk_size]
            # Split chunk into sentences
            chunk_sentences = sent_tokenize(chunk)
            for sentence in chunk_sentences:
                if len(sentence.strip()) > 0:  # Skip empty sentences
                    try:
                        result = sentiment_analyzer(sentence)[0]
                        sentiments.append({
                            'Sentence': sentence,
                            'Sentiment': result['label'],
                            'Score': result['score']
                        })
                    except Exception as e:
                        st.warning(f"Could not analyze sentence: {sentence[:50]}... Error: {str(e)}")
                        continue
    
    if sentiments:
        sentiment_df = pd.DataFrame(sentiments)
        st.dataframe(sentiment_df)
        
        # Sentiment Distribution
        st.write("#### Sentiment Distribution")
        sentiment_counts = sentiment_df['Sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title='Distribution of Sentiments'
        )
        st.plotly_chart(fig)
        
        # Sentiment Score Distribution
        st.write("#### Sentiment Score Distribution")
        fig = px.histogram(
            sentiment_df,
            x='Score',
            color='Sentiment',
            title='Distribution of Sentiment Scores',
            marginal='box'
        )
        st.plotly_chart(fig)
    else:
        st.warning("No sentiments could be analyzed. The text might be too short or contain unsupported characters.")

def handle_nlp(data):
    st.subheader("NLP Model Configuration")
    
    # Text preprocessing options
    st.write("### Text Preprocessing")
    text_column = st.selectbox("Select text column", data.columns)
    
    if text_column:
        # Initialize NLP components
        preprocessor = TextPreprocessor()
        sentiment_analyzer = SentimentAnalyzer()
        word_freq_analyzer = WordFrequencyAnalyzer()
        
        # Display sample text
        st.write("Sample text from the dataset:")
        st.write(data[text_column].head())
        
        # Text cleaning options
        st.write("#### Text Cleaning Options")
        remove_numbers = st.checkbox("Remove numbers", value=True)
        remove_special_chars = st.checkbox("Remove special characters", value=True)
        lowercase = st.checkbox("Convert to lowercase", value=True)
        
        # Tokenization options
        st.write("#### Tokenization Options")
        tokenization_method = st.selectbox(
            "Tokenization Method",
            ["Word", "Sentence", "Both"]
        )
        
        # Stopword options
        st.write("#### Stopword Options")
        remove_stopwords = st.checkbox("Remove stopwords", value=True)
        custom_stopwords = st.text_area(
            "Custom stopwords (one per line)",
            value=""
        )
        
        # Word normalization
        st.write("#### Word Normalization")
        normalization_method = st.selectbox(
            "Normalization Method",
            ["Lemmatization", "Stemming"]
        )
        
        # Process text
        if st.button("Process Text"):
            with st.spinner("Processing text..."):
                # Basic text cleaning
                text = data[text_column].astype(str)
                cleaned_text = preprocessor.clean_text(
                    text,
                    remove_numbers=remove_numbers,
                    remove_special_chars=remove_special_chars,
                    lowercase=lowercase
                )
                
                # Tokenization
                tokens = preprocessor.tokenize(cleaned_text, method=tokenization_method.lower())
                
                # Stopword removal
                if remove_stopwords:
                    tokens = preprocessor.remove_stopwords(tokens, custom_stopwords)
                
                # Word normalization
                tokens = preprocessor.normalize_words(tokens, method=normalization_method.lower())
                
                # Display results
                st.write("### Processed Text")
                if tokenization_method in ["Word", "Both"]:
                    st.write("First 5 documents (word tokens):")
                    st.write(tokens.head())
                
                # Word frequency analysis
                st.write("### Word Frequency Analysis")
                if tokenization_method in ["Word", "Both"]:
                    word_freq = word_freq_analyzer.analyze(tokens)
                    
                    # Display top 20 most common words
                    st.write("Top 20 most common words:")
                    fig = word_freq_analyzer.plot_word_frequency(word_freq)
                    st.plotly_chart(fig)
                    
                    # Word cloud
                    st.write("### Word Cloud")
                    fig = word_freq_analyzer.generate_word_cloud(word_freq)
                    st.pyplot(fig)
                
                # Sentiment analysis
                st.write("### Sentiment Analysis")
                sentiments = sentiment_analyzer.analyze(cleaned_text)
                
                if not sentiments.empty:
                    st.write("Sentiment distribution:")
                    sentiment_counts = sentiments['label'].value_counts()
                    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment Distribution')
                    st.plotly_chart(fig)
                    
                    st.write("Sentiment scores:")
                    fig = px.histogram(sentiments, x='score', color='label', title='Sentiment Score Distribution')
                    st.plotly_chart(fig)
                else:
                    st.warning("No sentiments could be analyzed. The text might be too short or contain unsupported characters.")
                
                # Save processed data
                if st.button("Save Processed Data"):
                    processed_data = pd.DataFrame({
                        'original_text': data[text_column],
                        'processed_tokens': tokens
                    })
                    processed_data.to_csv('processed_text_data.csv', index=False)
                    st.success("Processed data saved successfully!")

if __name__ == "__main__":
    main() 