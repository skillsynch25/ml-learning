import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import math

class CNNModel:
    def __init__(self, input_shape, num_classes):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return history

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, bidirectional):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class NLPModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def preprocess_text(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    def get_embeddings(self, texts):
        inputs = self.preprocess_text(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

class Autoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.encoder = models.Sequential([
            layers.Dense(encoding_dim, activation='relu', input_shape=(input_dim,))
        ])
        self.decoder = models.Sequential([
            layers.Dense(input_dim, activation='sigmoid')
        ])
        self.autoencoder = models.Sequential([self.encoder, self.decoder])
        
    def train(self, X_train, epochs=10, batch_size=32):
        self.autoencoder.compile(optimizer='adam', loss='mse')
        history = self.autoencoder.fit(X_train, X_train,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     shuffle=True)
        return history

class GAN:
    def __init__(self, latent_dim, img_shape):
        self.generator = self.build_generator(latent_dim, img_shape)
        self.discriminator = self.build_discriminator(img_shape)
        self.combined = self.build_combined()
        
    def build_generator(self, latent_dim, img_shape):
        model = models.Sequential([
            layers.Dense(256, input_dim=latent_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(momentum=0.8),
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(momentum=0.8),
            layers.Dense(np.prod(img_shape), activation='tanh'),
            layers.Reshape(img_shape)
        ])
        return model
    
    def build_discriminator(self, img_shape):
        model = models.Sequential([
            layers.Flatten(input_shape=img_shape),
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def build_combined(self):
        self.discriminator.trainable = False
        model = models.Sequential([self.generator, self.discriminator])
        return model
    
    def train(self, X_train, epochs=100, batch_size=32):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            
        return self.generator, self.discriminator 

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