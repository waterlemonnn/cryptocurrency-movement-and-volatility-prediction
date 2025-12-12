import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional, Multiply, Permute, Reshape, Flatten
from tensorflow.keras.optimizers import Adam

def attention_layer(inputs, time_steps):
    # Input shape: (batch_size, time_steps, features)
    a = Permute((2, 1))(inputs)
    a = Reshape((inputs.shape[2], time_steps))(a)
    a = Dense(time_steps, activation='softmax')(a)
    
    a_probs = Permute((2, 1), name='attention_vec')(a)
    
    # Output: (batch_size, time_steps, features)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def build_attention_model(input_shape, lr=0.0005):
    inputs = Input(shape=input_shape)
    
    # LSTM Layers
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # Attention Mechanism
    attention_mul = attention_layer(lstm_out, input_shape[0])
    
    # Flatten
    x = Flatten()(attention_mul)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output Heads
    # Direction (Binary Classification: 0 or 1)
    dir_branch = Dense(32, activation='relu')(x)
    output_dir = Dense(1, activation='sigmoid', name='out_dir')(dir_branch)
    
    # Volatility (Regression: Continuous Value)
    vol_branch = Dense(32, activation='relu')(x)
    output_vol = Dense(1, activation='relu', name='out_vol')(vol_branch)
    
    model = Model(inputs=inputs, outputs=[output_dir, output_vol])
    
    # Compile Model
    losses = {'out_dir': 'binary_crossentropy', 'out_vol': 'mse'}
    loss_weights = {'out_dir': 50.0, 'out_vol': 1.0}
    
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=losses,
                  loss_weights=loss_weights,
                  metrics=[['accuracy'], ['mae']])
    
    return model