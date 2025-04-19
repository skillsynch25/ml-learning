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
        
        # Model selection
        model_types = ["Linear Regression", "Random Forest", "XGBoost", "LightGBM"]
        if CATBOOST_AVAILABLE:
            model_types.append("CatBoost")
        
        model_type = st.sidebar.selectbox(
            "Select Model",
            model_types
        )
        
        # Model parameters
        st.sidebar.subheader("Model Parameters")
        if model_type == "Linear Regression":
            fit_intercept = st.sidebar.checkbox("Fit Intercept", value=True)
            normalize = st.sidebar.checkbox("Normalize", value=False)
            
        elif model_type == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)
            
        elif model_type == "XGBoost":
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1)
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 6)
            
        elif model_type == "LightGBM":
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1)
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 6)
            
        elif model_type == "CatBoost" and CATBOOST_AVAILABLE:
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1)
            iterations = st.sidebar.slider("Number of Trees", 10, 200, 100)
            depth = st.sidebar.slider("Max Depth", 1, 20, 6)
        
        # Train-test split
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        
        # Train model
        with st.spinner("Training model..."):
            if model_type == "Linear Regression":
                model = train_linear_regression(X_train, y_train, fit_intercept, normalize)
            elif model_type == "Random Forest":
                model = train_random_forest(X_train, y_train, n_estimators, max_depth)
            elif model_type == "XGBoost":
                model = train_xgboost(X_train, y_train, learning_rate, n_estimators, max_depth)
            elif model_type == "LightGBM":
                model = train_lightgbm(X_train, y_train, learning_rate, n_estimators, max_depth)
            elif model_type == "CatBoost" and CATBOOST_AVAILABLE:
                model = train_catboost(X_train, y_train, learning_rate, iterations, depth)
            
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
    
    # Display data types
    st.write("#### Data Types Overview")
    data_types = pd.DataFrame({
        'Column': data.columns,
        'Data Type': data.dtypes,
        'Unique Values': data.nunique(),
        'Missing Values': data.isnull().sum()
    })
    st.dataframe(data_types)
    
    # Select target column
    target_column = st.selectbox("Select target column", data.columns)
    
    # Select feature columns
    feature_columns = st.multiselect(
        "Select feature columns",
        [col for col in data.columns if col != target_column],
        default=[col for col in data.columns if col != target_column]
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
        
        if categorical_columns:
            st.write("Categorical columns detected:", categorical_columns)
            encoding_method = st.selectbox(
                "Select encoding method for categorical variables",
                ["One-Hot Encoding", "Label Encoding", "Ordinal Encoding"]
            )
            
            # Show sample of categorical data
            st.write("Sample of categorical data:")
            st.dataframe(data[categorical_columns].head())
        
        if numerical_columns:
            st.write("Numerical columns detected:", numerical_columns)
            # Show sample of numerical data
            st.write("Sample of numerical data:")
            st.dataframe(data[numerical_columns].head())
            
            # Show basic statistics
            st.write("Basic statistics for numerical columns:")
            st.dataframe(data[numerical_columns].describe())
        
        # Data Visualization
        st.write("#### Data Visualization")
        
        # Plot numerical distributions
        if numerical_columns:
            st.write("##### Numerical Distributions")
            for col in numerical_columns:
                fig = px.histogram(data, x=col, title=f'Distribution of {col}')
                st.plotly_chart(fig)
        
        # Plot categorical distributions
        if categorical_columns:
            st.write("##### Categorical Distributions")
            for col in categorical_columns:
                value_counts = data[col].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f'Distribution of {col}',
                    labels={'x': col, 'y': 'Count'}
                )
                st.plotly_chart(fig)
        
        # Correlation analysis for numerical columns
        if len(numerical_columns) > 1:
            st.write("##### Correlation Analysis")
            corr_matrix = data[numerical_columns].corr()
            fig = px.imshow(
                corr_matrix,
                title='Correlation Matrix',
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig)
        
        # Sequence Configuration
        st.write("#### Sequence Configuration")
        sequence_length = st.slider(
            "Sequence Length",
            min_value=1,
            max_value=100,
            value=10,
            key="sequence_length_slider"
        )
        
        # Data Preprocessing
        st.write("#### Data Preprocessing")
        
        # Handle categorical variables
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
                key="encoding_method_select"
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
        def create_sequences(data, sequence_length):
            sequences = []
            for i in range(len(data) - sequence_length):
                sequences.append(data[i:(i + sequence_length)])
            return np.array(sequences)
        
        # Convert input data to numpy array first
        X_array = np.array(X.values, dtype=np.float32)
        y_array = np.array(y.values, dtype=np.int64)  # Convert target to int64
        
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
        X_train = torch.FloatTensor(X_train)  # Keep input as float
        X_test = torch.FloatTensor(X_test)    # Keep input as float
        y_train = torch.FloatTensor(y_train).view(-1, 1)  # Reshape target to [batch_size, 1]
        y_test = torch.FloatTensor(y_test).view(-1, 1)    # Reshape target to [batch_size, 1]
        
        # Display data shapes and types
        st.write("#### Data Shapes and Types")
        data_info = pd.DataFrame({
            'Data': ['X_train', 'X_test', 'y_train', 'y_test'],
            'Shape': [str(X_train.shape), str(X_test.shape), str(y_train.shape), str(y_test.shape)],
            'Type': [str(X_train.dtype), str(X_test.dtype), str(y_train.dtype), str(y_test.dtype)]
        })
        st.dataframe(data_info)
        
        # Display target variable statistics
        st.write("#### Target Variable Statistics")
        target_stats = pd.DataFrame({
            'Statistic': ['Min', 'Max', 'Mean', 'Std'],
            'Value': [
                float(y_train.min()),
                float(y_train.max()),
                float(y_train.mean()),
                float(y_train.std())
            ]
        })
        st.dataframe(target_stats)
        
        # Display tensor information
        st.write("Tensor Information:")
        tensor_info = pd.DataFrame({
            'Tensor': ['X_train', 'X_test', 'y_train', 'y_test'],
            'Shape': [str(X_train.shape), str(X_test.shape), str(y_train.shape), str(y_test.shape)],
            'Type': [str(X_train.dtype), str(X_test.dtype), str(y_train.dtype), str(y_test.dtype)]
        })
        st.dataframe(tensor_info)
        
        # Train model
        history = None
        model = None  # Initialize model variable
        with st.spinner("Training Model..."):
            try:
                # Convert training parameters to proper types
                epochs = int(params['epochs'])
                batch_size = int(params['batch_size'])
                
                # Create LSTM model
                model = LSTMModel(
                    input_dim=int(X_train.shape[2]),
                    hidden_dim=int(params['lstm_units']),
                    num_layers=int(params['num_lstm_layers']),
                    output_dim=1,  # For regression
                    dropout=params['lstm_dropout'],
                    bidirectional=params['lstm_bidirectional']
                )
                
                # Display model architecture details
                st.write("### Model Architecture Details")
                st.write(f"Input Dimension: {X_train.shape[2]} features")
                st.write(f"Hidden Dimension: {params['lstm_units']} units")
                st.write(f"Number of LSTM Layers: {params['num_lstm_layers']}")
                st.write(f"Bidirectional: {'Yes' if params['lstm_bidirectional'] else 'No'}")
                st.write(f"Dropout Rate: {params['lstm_dropout']}")
                
                # Calculate and display model parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                st.write(f"Total Parameters: {total_params:,}")
                st.write(f"Trainable Parameters: {trainable_params:,}")
                
                # Train model with correct parameters
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
                
                # Training loop
                train_losses = []
                val_losses = []
                
                for epoch in range(epochs):
                    model.train()
                    epoch_loss = 0
                    
                    for i in range(0, len(X_train), batch_size):
                        batch_X = X_train[i:i+batch_size]
                        batch_y = y_train[i:i+batch_size]
                        
                        # Ensure inputs are tensors and of correct type
                        if not isinstance(batch_X, torch.Tensor):
                            batch_X = torch.FloatTensor(batch_X)
                        if not isinstance(batch_y, torch.Tensor):
                            batch_y = torch.FloatTensor(batch_y).view(-1, 1)  # Reshape target
                        
                        # Forward pass
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        
                        # Calculate loss
                        loss = criterion(outputs, batch_y)
                        
                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    # Validation
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_test)
                        val_loss = criterion(val_outputs, y_test)
                    
                    train_losses.append(epoch_loss / (len(X_train) // batch_size))
                    val_losses.append(val_loss.item())
                    
                    if (epoch + 1) % 5 == 0:
                        st.write(f"Epoch [{epoch+1}/{epochs}], "
                               f"Train Loss: {train_losses[-1]:.4f}, "
                               f"Val Loss: {val_losses[-1]:.4f}")
                
                # Display final training statistics
                st.write("### Training Statistics")
                st.write(f"Final Training Loss: {train_losses[-1]:.4f}")
                st.write(f"Final Validation Loss: {val_losses[-1]:.4f}")
                
                # Calculate and display additional metrics
                model.eval()
                with torch.no_grad():
                    train_preds = model(X_train)
                    test_preds = model(X_test)
                    
                    # Calculate R-squared score
                    train_ss_res = torch.sum((y_train - train_preds) ** 2)
                    train_ss_tot = torch.sum((y_train - torch.mean(y_train)) ** 2)
                    train_r2 = 1 - train_ss_res / train_ss_tot
                    
                    test_ss_res = torch.sum((y_test - test_preds) ** 2)
                    test_ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
                    test_r2 = 1 - test_ss_res / test_ss_tot
                    
                    st.write("### Model Performance Metrics")
                    st.write(f"Training R² Score: {train_r2.item():.4f}")
                    st.write(f"Testing R² Score: {test_r2.item():.4f}")
                    
                    # Calculate Mean Absolute Error
                    train_mae = torch.mean(torch.abs(y_train - train_preds))
                    test_mae = torch.mean(torch.abs(y_test - test_preds))
                    st.write(f"Training MAE: {train_mae.item():.4f}")
                    st.write(f"Testing MAE: {test_mae.item():.4f}")
                    
                    # Calculate Root Mean Squared Error
                    train_rmse = torch.sqrt(torch.mean((y_train - train_preds) ** 2))
                    test_rmse = torch.sqrt(torch.mean((y_test - test_preds) ** 2))
                    st.write(f"Training RMSE: {train_rmse.item():.4f}")
                    st.write(f"Testing RMSE: {test_rmse.item():.4f}")
                
                # Plot training history
                st.write("### Training History")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(train_losses, label='Training Loss')
                ax.plot(val_losses, label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training and Validation Loss')
                ax.legend()
                st.pyplot(fig)
                
                # Plot predictions vs actual values
                st.write("### Predictions vs Actual Values")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(y_test.cpu().numpy(), test_preds.cpu().numpy(), alpha=0.5)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Actual vs Predicted Values')
                st.pyplot(fig)
                
                # Display model summary
                st.write("### Model Summary")
                st.write("The LSTM model has been trained with the following characteristics:")
                st.write(f"- Input sequence length: {X_train.shape[1]}")
                st.write(f"- Number of features per time step: {X_train.shape[2]}")
                st.write(f"- Hidden state size: {params['lstm_units']}")
                st.write(f"- Number of LSTM layers: {params['num_lstm_layers']}")
                st.write(f"- Dropout rate: {params['lstm_dropout']}")
                st.write(f"- Bidirectional: {'Yes' if params['lstm_bidirectional'] else 'No'}")
                st.write(f"- Total parameters: {total_params:,}")
                st.write(f"- Training epochs: {epochs}")
                st.write(f"- Batch size: {batch_size}")
                st.write(f"- Learning rate: {params['learning_rate']}")
                
                # Display performance interpretation
                st.write("### Performance Interpretation")
                st.write("The model's performance can be interpreted as follows:")
                st.write(f"- R² Score: {test_r2.item():.4f} indicates that the model explains {test_r2.item()*100:.2f}% of the variance in the target variable")
                st.write(f"- RMSE: {test_rmse.item():.4f} represents the average prediction error in the same units as the target variable")
                st.write(f"- MAE: {test_mae.item():.4f} shows the average absolute difference between predictions and actual values")
                
                if test_r2.item() > 0.7:
                    st.success("The model shows strong predictive performance!")
                elif test_r2.item() > 0.5:
                    st.info("The model shows moderate predictive performance.")
                else:
                    st.warning("The model's predictive performance could be improved. Consider:")
                    st.write("- Increasing model complexity")
                    st.write("- Adding more features")
                    st.write("- Adjusting hyperparameters")
                    st.write("- Collecting more training data")
                
            except Exception as e:
                st.error(f"Error during training LSTM: {str(e)}")
                st.write("Please check your model parameters and data format.")
                model = None  # Reset model on error
        
        # Feature importance (if applicable)
        if model is not None and hasattr(model, 'get_feature_importance'):
            st.subheader("Feature Importance")
            importance = model.get_feature_importance(X_test)
            fig = px.bar(
                x=feature_columns,
                y=importance,
                title='Feature Importance'
            )
            st.plotly_chart(fig)
        
    else:
        st.warning("Please select both target and feature columns to proceed with model training.")

def handle_cnn(data, params):
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

def handle_rnn(data, params):
    st.subheader("RNN Model Configuration")
    
    # Data Preprocessing
    st.write("### Data Preprocessing")
    
    # Display data types
    st.write("#### Data Types Overview")
    data_types = pd.DataFrame({
        'Column': data.columns,
        'Data Type': data.dtypes,
        'Unique Values': data.nunique(),
        'Missing Values': data.isnull().sum()
    })
    st.dataframe(data_types)
    
    # Select target column
    target_column = st.selectbox("Select target column", data.columns)
    
    # Select feature columns
    feature_columns = st.multiselect(
        "Select feature columns",
        [col for col in data.columns if col != target_column],
        default=[col for col in data.columns if col != target_column]
    )
    
    # Data type handling
    st.write("#### Data Type Handling")
    categorical_columns = []
    numerical_columns = []
    
    for col in feature_columns:
        if data[col].dtype in ['object', 'category', 'string']:
            categorical_columns.append(col)
        else:
            numerical_columns.append(col)
    
    if categorical_columns:
        st.write("Categorical columns detected:", categorical_columns)
        encoding_method = st.selectbox(
            "Select encoding method for categorical variables",
            ["One-Hot Encoding", "Label Encoding", "Ordinal Encoding"]
        )
        
        # Show sample of categorical data
        st.write("Sample of categorical data:")
        st.dataframe(data[categorical_columns].head())
    
    if numerical_columns:
        st.write("Numerical columns detected:", numerical_columns)
        # Show sample of numerical data
        st.write("Sample of numerical data:")
        st.dataframe(data[numerical_columns].head())
        
        # Show basic statistics
        st.write("Basic statistics for numerical columns:")
        st.dataframe(data[numerical_columns].describe())
    
    # Data Visualization
    st.write("#### Data Visualization")
    
    # Plot numerical distributions
    if numerical_columns:
        st.write("##### Numerical Distributions")
        for col in numerical_columns:
            fig = px.histogram(data, x=col, title=f'Distribution of {col}')
            st.plotly_chart(fig)
    
    # Plot categorical distributions
    if categorical_columns:
        st.write("##### Categorical Distributions")
        for col in categorical_columns:
            value_counts = data[col].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution of {col}',
                labels={'x': col, 'y': 'Count'}
            )
            st.plotly_chart(fig)
    
    # Correlation analysis for numerical columns
    if len(numerical_columns) > 1:
        st.write("##### Correlation Analysis")
        corr_matrix = data[numerical_columns].corr()
        fig = px.imshow(
            corr_matrix,
            title='Correlation Matrix',
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig)
    
    # Sequence Configuration
    st.write("#### Sequence Configuration")
    sequence_length = st.slider(
        "Sequence Length",
        min_value=1,
        max_value=100,
        value=10,
        key="sequence_length_slider"
    )
    
    # Data Preprocessing
    st.write("#### Data Preprocessing")
    
    # Handle categorical variables
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
            key="encoding_method_select"
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
    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:(i + sequence_length)])
        return np.array(sequences)
    
    # Convert input data to numpy array first
    X_array = np.array(X.values, dtype=np.float32)
    y_array = np.array(y.values, dtype=np.int64)  # Convert target to int64
    
    # Create sequences
    X_sequences = create_sequences(X_array, sequence_length)
    y_sequences = y_array[sequence_length:]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, y_sequences, test_size=0.2, random_state=42
    )
    
    # Ensure proper shape for RNN input
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Convert to PyTorch tensors
    import torch
    X_train = torch.FloatTensor(X_train)  # Keep input as float
    X_test = torch.FloatTensor(X_test)    # Keep input as float
    y_train = torch.FloatTensor(y_train).view(-1, 1)  # Reshape target to [batch_size, 1]
    y_test = torch.FloatTensor(y_test).view(-1, 1)    # Reshape target to [batch_size, 1]
    
    # Display data shapes and types
    st.write("#### Data Shapes and Types")
    data_info = pd.DataFrame({
        'Data': ['X_train', 'X_test', 'y_train', 'y_test'],
        'Shape': [str(X_train.shape), str(X_test.shape), str(y_train.shape), str(y_test.shape)],
        'Type': [str(X_train.dtype), str(X_test.dtype), str(y_train.dtype), str(y_test.dtype)]
    })
    st.dataframe(data_info)
    
    # Display target variable statistics
    st.write("#### Target Variable Statistics")
    target_stats = pd.DataFrame({
        'Statistic': ['Min', 'Max', 'Mean', 'Std'],
        'Value': [
            float(y_train.min()),
            float(y_train.max()),
            float(y_train.mean()),
            float(y_train.std())
        ]
    })
    st.dataframe(target_stats)
    
    # Display tensor information
    st.write("Tensor Information:")
    tensor_info = pd.DataFrame({
        'Tensor': ['X_train', 'X_test', 'y_train', 'y_test'],
        'Shape': [str(X_train.shape), str(X_test.shape), str(y_train.shape), str(y_test.shape)],
        'Type': [str(X_train.dtype), str(X_test.dtype), str(y_train.dtype), str(y_test.dtype)]
    })
    st.dataframe(tensor_info)
    
    # Train model
    history = None
    model = None  # Initialize model variable
    with st.spinner("Training RNN..."):
        try:
            # Convert training parameters to proper types
            epochs = int(params['epochs'])
            batch_size = int(params['batch_size'])
            
            # Create RNN model
            rnn_model = RNNModel(
                input_dim=int(X_train.shape[2]),
                hidden_dim=int(params['hidden_units']),
                num_layers=int(params['num_rnn_layers']),
                num_classes=int(len(np.unique(y)))  # Number of unique classes
            )
            
            # Train model with correct parameters
            history = rnn_model.train(
                X_train,  # First positional argument
                y_train,  # Second positional argument
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Display training progress
            st.write("Training completed successfully!")
            st.write(f"Final loss: {history.history['loss'][-1]:.4f}")
            if 'val_loss' in history.history:
                st.write(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            st.write("Please check your model parameters and data format.")
            model = None  # Reset model on error
    
    # Visualizations
    st.subheader("Training Results")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot (if available)
    if 'accuracy' in history.history:
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
    
    st.pyplot(fig)
    
    # Model predictions
    st.subheader("Model Predictions")
    y_pred = rnn_model.predict(X_test)
    
    # Plot predictions vs actual
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y_test.flatten(),
        mode='lines',
        name='Actual'
    ))
    fig.add_trace(go.Scatter(
        y=y_pred.flatten(),
        mode='lines',
        name='Predicted'
    ))
    fig.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Sample',
        yaxis_title='Value'
    )
    st.plotly_chart(fig)
    
    # Feature importance (if applicable)
    if model is not None and hasattr(rnn_model, 'get_feature_importance'):
        st.subheader("Feature Importance")
        importance = rnn_model.get_feature_importance(X_test)
        fig = px.bar(
            x=feature_columns,
            y=importance,
            title='Feature Importance'
        )
        st.plotly_chart(fig)
    

def handle_lstm(data, params):
    st.subheader("LSTM Model Implementation")
    
    # Initialize model with correct parameters
    input_dim = X_train.shape[2]
    hidden_dim = params['lstm_units']
    num_layers = params['num_lstm_layers']  # Use the correct parameter name
    output_dim = 1  # For regression
    dropout = params['lstm_dropout']
    bidirectional = params['lstm_bidirectional']
    
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout, bidirectional)
    
    # Step 6: Training Configuration
    st.write("### Step 6: Training Configuration")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Step 7: Training Process
    st.write("### Step 7: Training Process")
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
            
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0
                
                for batch_X, batch_y in train_loader:
                    # Ensure inputs are tensors and of correct type
                    if not isinstance(batch_X, torch.Tensor):
                        batch_X = torch.FloatTensor(batch_X)
                    if not isinstance(batch_y, torch.Tensor):
                        batch_y = torch.FloatTensor(batch_y).view(-1, 1)  # Reshape target
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    # Calculate loss
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    # Ensure validation data is tensors
                    if not isinstance(X_test, torch.Tensor):
                        X_test = torch.FloatTensor(X_test)
                    if not isinstance(y_test, torch.Tensor):
                        y_test = torch.FloatTensor(y_test).view(-1, 1)    # Reshape target to [batch_size, 1]
                    
                    val_outputs = model(X_test)
                    val_loss = criterion(val_outputs, y_test)
                
                train_losses.append(epoch_loss / len(train_loader))
                val_losses.append(val_loss.item())
                
                if (epoch + 1) % 5 == 0:
                    st.write(f"Epoch [{epoch+1}/{num_epochs}], "
                           f"Train Loss: {train_losses[-1]:.4f}, "
                           f"Val Loss: {val_losses[-1]:.4f}")
            
            # Plot training history
            st.write("### Training History")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(train_losses, label='Training Loss')
            ax.plot(val_losses, label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            st.pyplot(fig)
            
            # Step 8: Model Evaluation
            st.write("### Step 8: Model Evaluation")
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                mse = criterion(y_pred, y_test)
                rmse = torch.sqrt(mse)
                
                st.write("Evaluation Metrics:")
                st.write(f"Mean Squared Error (MSE): {mse.item():.4f}")
                st.write(f"Root Mean Squared Error (RMSE): {rmse.item():.4f}")
                
                # Plot predictions vs actual
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(y_test.numpy(), y_pred.numpy(), alpha=0.5)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Actual vs Predicted Values')
                st.pyplot(fig)
            
            # Step 9: Model Saving
            st.write("### Step 9: Model Saving")
            if st.button("Save Model"):
                torch.save(model.state_dict(), 'lstm_model.pth')
                st.success("Model saved successfully!")

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

def handle_transformer(data, params):
    st.subheader("Transformer Model Configuration")
    
    # Model parameters
    input_dim = st.sidebar.number_input("Input Dimension", min_value=1, value=10)
    hidden_dim = st.sidebar.number_input("Hidden Dimension", min_value=1, value=64)
    num_layers = st.sidebar.number_input("Number of Layers", min_value=1, value=2)
    num_classes = st.sidebar.number_input("Number of Classes", min_value=2, value=10)
    
    # Create and train model
    if st.sidebar.button("Train Transformer"):
        transformer_model = TransformerModel(input_dim, hidden_dim, num_layers, num_classes)
        
        # Training parameters
        epochs = st.sidebar.slider("Epochs", 1, 50, 10)
        batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
        
        # Train model
        with st.spinner("Training Transformer..."):
            transformer_model.train(data, epochs=epochs, batch_size=batch_size)
            
            # Visualizations
            st.subheader("Model Architecture")
            st.image(plot_model_architecture(transformer_model.model))

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

if __name__ == "__main__":
    main() 