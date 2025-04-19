import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

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

class RNNModel:
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        self.model = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = torch.FloatTensor(X_train[i:i+batch_size])
                batch_y = torch.LongTensor(y_train[i:i+batch_size])
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self.model

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