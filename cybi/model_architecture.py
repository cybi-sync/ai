import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import logging
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax, AdamW
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, 
    BatchNormalization, Dropout, Input, Concatenate, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D, Attention, Add
)

logger = logging.getLogger(__name__)

class ResidualBlock(layers.Layer):
    """Custom residual block for time series data."""
    def __init__(self, filters, kernel_size, dilation_rate=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv1D(
            filters, kernel_size, padding='same', 
            dilation_rate=dilation_rate,
            activation='relu'
        )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(
            filters, kernel_size, padding='same', 
            dilation_rate=dilation_rate
        )
        self.bn2 = layers.BatchNormalization()
        
        # Skip connection if input shape doesn't match output shape
        self.skip_connection = layers.Conv1D(filters, 1, padding='same') if True else None
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Apply skip connection
        if self.skip_connection is not None:
            skip = self.skip_connection(inputs)
        else:
            skip = inputs
            
        return layers.ReLU()(x + skip)

class TransformerBlock(layers.Layer):
    """Transformer block for sequential data processing."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TimeDistributedAttention(layers.Layer):
    """Applies attention over time dimension."""
    def __init__(self, units):
        super(TimeDistributedAttention, self).__init__()
        self.w1 = layers.Dense(units)
        self.w2 = layers.Dense(1)
        
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        x = tf.nn.tanh(self.w1(inputs))
        x = self.w2(x)
        x = tf.nn.softmax(x, axis=1)
        
        # Weighted sum using the attention weights
        return tf.reduce_sum(inputs * x, axis=1)

class CYBIModelBuilder:
    """Builds complex deep learning architectures for CYBI smartshoe data analysis."""
    
    def __init__(self, config=None):
        self.config = config or {
            'weight_prediction': {
                'architecture': 'hybrid',  # hybrid, cnn, lstm, transformer
                'input_shape': (100, 50),  # (time_steps, features)
                'lstm_units': [64, 128],
                'cnn_filters': [64, 128, 256],
                'cnn_kernel_sizes': [3, 5, 7],
                'dense_units': [512, 256, 128, 64],
                'dropout_rate': 0.3,
                'l2_reg': 0.001,
                'use_attention': True,
                'transformer_heads': 8,
                'transformer_dim': 256,
                'transformer_layers': 4,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'optimizer': 'adam',  # adam, sgd, rmsprop
                'early_stopping_patience': 15,
                'reduce_lr_patience': 5,
                'use_batch_norm': True
            },
            'health_prediction': {
                'architecture': 'multi_stream',  # multi_stream, ensemble, transformer
                'input_shape': (200, 75),  # (time_steps, features)
                'stream_architectures': ['cnn', 'lstm', 'transformer'],
                'lstm_units': [128, 256],
                'gru_units': [128, 256],
                'cnn_filters': [64, 128, 256, 512],
                'cnn_kernel_sizes': [3, 5, 9, 15],
                'dense_units': [1024, 512, 256, 128],
                'dropout_rate': 0.4,
                'l2_reg': 0.0015,
                'use_attention': True,
                'transformer_heads': 12,
                'transformer_dim': 512,
                'transformer_layers': 6,
                'learning_rate': 0.0005,
                'batch_size': 16,
                'epochs': 150,
                'optimizer': 'adamw',  # adam, adamw, sgd, rmsprop
                'early_stopping_patience': 20,
                'reduce_lr_patience': 8,
                'use_batch_norm': True,
                'task_type': 'multi_class',  # binary, multi_class, multi_label
                'num_classes': 15  # Number of health conditions to detect
            }
        }
    
    def _get_optimizer(self, config_section):
        """Get the specified optimizer with the configured learning rate."""
        optimizer_name = self.config[config_section]['optimizer'].lower()
        lr = self.config[config_section]['learning_rate']
        
        if optimizer_name == 'adam':
            return Adam(learning_rate=lr)
        elif optimizer_name == 'adamw':
            return AdamW(learning_rate=lr, weight_decay=0.001)
        elif optimizer_name == 'sgd':
            return SGD(learning_rate=lr, momentum=0.9, nesterov=True)
        elif optimizer_name == 'rmsprop':
            return RMSprop(learning_rate=lr)
        else:
            logger.warning(f"Unknown optimizer {optimizer_name}, using Adam as default")
            return Adam(learning_rate=lr)
    
    def _get_callbacks(self, config_section, model_name):
        """Get training callbacks based on configuration."""
        es_patience = self.config[config_section]['early_stopping_patience']
        lr_patience = self.config[config_section]['reduce_lr_patience']
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=es_patience, 
                restore_best_weights=True, 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=lr_patience, 
                min_lr=1e-6, 
                verbose=1
            ),
            ModelCheckpoint(
                f"./models/{model_name}_best.h5", 
                save_best_only=True, 
                monitor='val_loss'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'./logs/{model_name}',
                update_freq='epoch',
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def build_weight_prediction_model(self):
        """Build a complex model for weight prediction from smartshoe data."""
        config = self.config['weight_prediction']
        input_shape = config['input_shape']
        architecture = config['architecture']
        
        # Define model input
        inputs = Input(shape=input_shape)
        
        # Apply different architectures based on configuration
        if architecture == 'cnn':
            x = self._build_cnn_architecture(inputs, config)
        elif architecture == 'lstm':
            x = self._build_rnn_architecture(inputs, config)
        elif architecture == 'transformer':
            x = self._build_transformer_architecture(inputs, config)
        elif architecture == 'hybrid':
            x = self._build_hybrid_architecture(inputs, config)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Final prediction layers with appropriate regularization
        x = Dense(
            config['dense_units'][-1], 
            activation='relu',
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        )(x)
        
        if config['use_batch_norm']:
            x = BatchNormalization()(x)
            
        x = Dropout(config['dropout_rate'])(x)
        
        # Output layer for weight prediction (regression task)
        outputs = Dense(1, activation='linear', name='weight_prediction')(x)
        
        # Create and compile the model
        model = keras.Model(inputs=inputs, outputs=outputs, name='weight_predictor')
        
        # Get optimizer
        optimizer = self._get_optimizer('weight_prediction')
        
        # Compile with appropriate loss for regression
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()]
        )
        
        return model
    
    def build_health_prediction_model(self):
        """Build a complex model for health condition prediction."""
        config = self.config['health_prediction']
        input_shape = config['input_shape']
        architecture = config['architecture']
        
        # Define model input
        inputs = Input(shape=input_shape)
        
        # Apply different architectures based on configuration
        if architecture == 'multi_stream':
            x = self._build_multi_stream_architecture(inputs, config)
        elif architecture == 'ensemble':
            # For ensemble, we'll return a list of models later
            return self._build_ensemble_models(config)
        elif architecture == 'transformer':
            x = self._build_transformer_architecture(inputs, config, is_health=True)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Final prediction layers
        x = Dense(
            config['dense_units'][-1], 
            activation='relu',
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        )(x)
        
        if config['use_batch_norm']:
            x = BatchNormalization()(x)
            
        x = Dropout(config['dropout_rate'])(x)
        
        # Output layer based on task type
        task_type = config['task_type']
        num_classes = config['num_classes']
        
        if task_type == 'binary':
            outputs = Dense(1, activation='sigmoid', name='health_prediction')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        
        elif task_type == 'multi_class':
            outputs = Dense(num_classes, activation='softmax', name='health_prediction')(x)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        
        elif task_type == 'multi_label':
            outputs = Dense(num_classes, activation='sigmoid', name='health_prediction')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        # Create and compile the model
        model = keras.Model(inputs=inputs, outputs=outputs, name='health_predictor')
        
        # Get optimizer
        optimizer = self._get_optimizer('health_prediction')
        
        # Compile with appropriate loss
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def _build_cnn_architecture(self, inputs, config):
        """Build a CNN-based feature extraction architecture."""
        x = inputs
        l2_reg = config['l2_reg']
        use_bn = config['use_batch_norm']
        dropout_rate = config['dropout_rate']
        
        for i, (filters, kernel_size) in enumerate(zip(
            config['cnn_filters'], 
            config['cnn_kernel_sizes']
        )):
            # Residual block with increasing dilation rates
            dilation_rate = 2**min(i, 3)  # Cap dilation at 2^3=8
            res_block = ResidualBlock(filters, kernel_size, dilation_rate)
            x = res_block(x)
            
            if use_bn:
                x = BatchNormalization()(x)
                
            # Add pooling every other layer
            if i % 2 == 1:
                x = MaxPooling1D(pool_size=2)(x)
                
            x = Dropout(dropout_rate)(x)
        
        # Global pooling to handle variable length inputs
        x = GlobalAveragePooling1D()(x)
        
        return x
    
    def _build_rnn_architecture(self, inputs, config):
        """Build an RNN-based (LSTM/GRU) feature extraction architecture."""
        x = inputs
        use_attention = config['use_attention']
        l2_reg = config['l2_reg']
        
        # First apply 1D convolution for feature extraction
        x = Conv1D(
            filters=64, 
            kernel_size=5, 
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        
        # Stack Bidirectional LSTM layers
        for i, units in enumerate(config['lstm_units']):
            return_sequences = i < len(config['lstm_units']) - 1 or use_attention
            
            x = Bidirectional(
                LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=config['dropout_rate'],
                    recurrent_dropout=config['dropout_rate']/2,
                    kernel_regularizer=regularizers.l2(l2_reg)
                )
            )(x)
            
            if config['use_batch_norm']:
                x = layers.TimeDistributed(BatchNormalization())(x)
        
        # Apply attention if configured
        if use_attention:
            attention_layer = TimeDistributedAttention(units=config['lstm_units'][-1])
            x = attention_layer(x)
        
        return x
    
    def _build_transformer_architecture(self, inputs, config, is_health=False):
        """Build a Transformer-based feature extraction architecture."""
        x = inputs
        
        # Get transformer configuration based on whether it's for weight or health
        if is_health:
            transformer_dim = config['transformer_dim']
            num_heads = config['transformer_heads'] 
            num_layers = config['transformer_layers']
        else:
            transformer_dim = config['transformer_dim']
            num_heads = config['transformer_heads']
            num_layers = config['transformer_layers']
        
        # First apply 1D convolution to map to transformer dimension
        x = Conv1D(
            filters=transformer_dim,
            kernel_size=1,
            padding='same'
        )(x)
        
        # Add positional encoding
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_encoding = self._positional_encoding(positions, transformer_dim)
        x = x + pos_encoding
        
        # Stack transformer blocks
        for _ in range(num_layers):
            transformer_block = TransformerBlock(
                embed_dim=transformer_dim,
                num_heads=num_heads,
                ff_dim=transformer_dim * 4,
                dropout=config['dropout_rate']
            )
            x = transformer_block(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        return x
    
    def _positional_encoding(self, positions, d_model):
        """Calculate positional encoding for transformer."""
        angle_rates = 1 / np.power(10000, (2 * (tf.range(d_model, dtype=tf.float32) // 2)) / d_model)
        angle_rads = tf.cast(positions, tf.float32) * angle_rates
        
        # Apply sin to even indices, cos to odd indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        # Alternate sin and cos
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def _build_hybrid_architecture(self, inputs, config):
        """Build a hybrid architecture combining CNN, RNN, and possibly transformer."""
        # CNN branch
        cnn_output = self._build_cnn_architecture(inputs, config)
        
        # RNN branch
        rnn_output = self._build_rnn_architecture(inputs, config)
        
        # Transformer branch (if configured with enough capacity)
        if len(config['dense_units']) > 2 and config.get('use_transformer_branch', False):
            transformer_output = self._build_transformer_architecture(inputs, config)
            
            # Combine all three branches
            combined = Concatenate()([cnn_output, rnn_output, transformer_output])
        else:
            # Combine CNN and RNN branches only
            combined = Concatenate()([cnn_output, rnn_output])
        
        # Add fusion layers
        x = Dense(
            config['dense_units'][0], 
            activation='relu', 
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        )(combined)
        
        if config['use_batch_norm']:
            x = BatchNormalization()(x)
            
        x = Dropout(config['dropout_rate'])(x)
        
        return x
    
    def _build_multi_stream_architecture(self, inputs, config):
        """Build a multi-stream architecture for health prediction."""
        stream_outputs = []
        
        # Process each stream based on the configured architectures
        for stream_arch in config['stream_architectures']:
            if stream_arch == 'cnn':
                stream_output = self._build_cnn_architecture(inputs, config)
            elif stream_arch == 'lstm':
                stream_output = self._build_rnn_architecture(inputs, config)
            elif stream_arch == 'gru':
                # Similar to LSTM but with GRU cells
                stream_output = self._build_gru_architecture(inputs, config)
            elif stream_arch == 'transformer':
                stream_output = self._build_transformer_architecture(inputs, config, is_health=True)
            else:
                raise ValueError(f"Unknown stream architecture: {stream_arch}")
            
            stream_outputs.append(stream_output)
        
        # If only one stream, return it directly
        if len(stream_outputs) == 1:
            return stream_outputs[0]
        
        # Combine all streams
        combined = Concatenate()(stream_outputs)
        
        # Add fusion layers
        x = Dense(
            config['dense_units'][0], 
            activation='relu', 
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        )(combined)
        
        if config['use_batch_norm']:
            x = BatchNormalization()(x)
            
        x = Dropout(config['dropout_rate'])(x)
        
        # Add a second fusion layer
        x = Dense(
            config['dense_units'][1], 
            activation='relu', 
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        )(x)
        
        if config['use_batch_norm']:
            x = BatchNormalization()(x)
            
        x = Dropout(config['dropout_rate'])(x)
        
        return x
    
    def _build_gru_architecture(self, inputs, config):
        """Build a GRU-based feature extraction architecture."""
        x = inputs
        use_attention = config['use_attention']
        l2_reg = config['l2_reg']
        
        # First apply 1D convolution for feature extraction
        x = Conv1D(
            filters=64, 
            kernel_size=5, 
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        
        # Stack Bidirectional GRU layers
        for i, units in enumerate(config['gru_units']):
            return_sequences = i < len(config['gru_units']) - 1 or use_attention
            
            x = Bidirectional(
                GRU(
                    units,
                    return_sequences=return_sequences,
                    dropout=config['dropout_rate'],
                    recurrent_dropout=config['dropout_rate']/2,
                    kernel_regularizer=regularizers.l2(l2_reg)
                )
            )(x)
            
            if config['use_batch_norm']:
                x = layers.TimeDistributed(BatchNormalization())(x)
        
        # Apply attention if configured
        if use_attention:
            attention_layer = TimeDistributedAttention(units=config['gru_units'][-1])
            x = attention_layer(x)
        
        return x
    
    def _build_ensemble_models(self, config):
        """Build an ensemble of models for health prediction."""
        models = []
        input_shape = config['input_shape']
        
        # Create a CNN-based model
        cnn_input = Input(shape=input_shape)
        cnn_output = self._build_cnn_architecture(cnn_input, config)
        cnn_output = Dense(
            config['dense_units'][-1], 
            activation='relu',
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        )(cnn_output)
        
        if config['use_batch_norm']:
            cnn_output = BatchNormalization()(cnn_output)
            
        cnn_output = Dropout(config['dropout_rate'])(cnn_output)
        
        # Output layer based on task type
        if config['task_type'] == 'binary':
            cnn_pred = Dense(1, activation='sigmoid', name='health_prediction')(cnn_output)
        elif config['task_type'] == 'multi_class':
            cnn_pred = Dense(config['num_classes'], activation='softmax', name='health_prediction')(cnn_output)
        elif config['task_type'] == 'multi_label':
            cnn_pred = Dense(config['num_classes'], activation='sigmoid', name='health_prediction')(cnn_output)
        
        cnn_model = keras.Model(inputs=cnn_input, outputs=cnn_pred, name='cnn_health_predictor')
        models.append(cnn_model)
        
        # Create an RNN-based model
        rnn_input = Input(shape=input_shape)
        rnn_output = self._build_rnn_architecture(rnn_input, config)
        rnn_output = Dense(
            config['dense_units'][-1], 
            activation='relu',
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        )(rnn_output)
        
        if config['use_batch_norm']:
            rnn_output = BatchNormalization()(rnn_output)
            
        rnn_output = Dropout(config['dropout_rate'])(rnn_output)
        
        # Output layer based on task type
        if config['task_type'] == 'binary':
            rnn_pred = Dense(1, activation='sigmoid', name='health_prediction')(rnn_output)
        elif config['task_type'] == 'multi_class':
            rnn_pred = Dense(config['num_classes'], activation='softmax', name='health_prediction')(rnn_output)
        elif config['task_type'] == 'multi_label':
            rnn_pred = Dense(config['num_classes'], activation='sigmoid', name='health_prediction')(rnn_output)
        
        rnn_model = keras.Model(inputs=rnn_input, outputs=rnn_pred, name='rnn_health_predictor')
        models.append(rnn_model)
        
        # Create a transformer-based model
        transformer_input = Input(shape=input_shape)
        transformer_output = self._build_transformer_architecture(transformer_input, config, is_health=True)
        transformer_output = Dense(
            config['dense_units'][-1], 
            activation='relu',
            kernel_regularizer=regularizers.l2(config['l2_reg'])
        )(transformer_output)
        
        if config['use_batch_norm']:
            transformer_output = BatchNormalization()(transformer_output)
            
        transformer_output = Dropout(config['dropout_rate'])(transformer_output)
        
        # Output layer based on task type
        if config['task_type'] == 'binary':
            transformer_pred = Dense(1, activation='sigmoid', name='health_prediction')(transformer_output)
        elif config['task_type'] == 'multi_class':
            transformer_pred = Dense(config['num_classes'], activation='softmax', name='health_prediction')(transformer_output)
        elif config['task_type'] == 'multi_label':
            transformer_pred = Dense(config['num_classes'], activation='sigmoid', name='health_prediction')(transformer_output)
        
        transformer_model = keras.Model(
            inputs=transformer_input, 
            outputs=transformer_pred, 
            name='transformer_health_predictor'
        )
        models.append(transformer_model)
        
        # Compile all models
        optimizer = self._get_optimizer('health_prediction')
        
        # Determine loss based on task type
        if config['task_type'] == 'binary':
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        elif config['task_type'] == 'multi_class':
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        elif config['task_type'] == 'multi_label':
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        
        for model in models:
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
        
        return models 