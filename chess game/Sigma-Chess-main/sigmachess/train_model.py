import os
import chess.engine
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Ensure output directory exists
os.makedirs("/kaggle/working/sigmachess_model", exist_ok=True)
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

# Import custom classes and models (assuming these are defined in accompanying files)
from classes import GameState, MCTS, ReplayBuffer
from models import sigmachess_network, create_model

# Harici Chess Engine Yolu
engine = chess.engine.SimpleEngine.popen_uci(r"/kaggle/input/test-1/stockfish-ubuntu-x86-64-avx2")

# Kaggle üzerinde model ağırlıklarının yolu
model_weights_path = "/kaggle/working/checkpoints/model_checkpoint.weights.h5"

def self_play(mcts, num_games=100):
    replay_buffer = ReplayBuffer(maxlen=500000)
    
    for game in range(num_games):
        state = GameState()
        temperature = 1.0 if game < 30 else 0.1
        states, policies, rewards = [], [], []
        player = np.random.choice([chess.WHITE, chess.BLACK])
        while not state.is_terminal():
            if state.board.turn == player:
                action_probs = mcts.run(state, temperature)
                
                states.append(state.get_current_state())
                policies.append(action_probs)
                action = np.random.choice(len(action_probs), p=action_probs)
                state.get_next_state(action)
            else:
                result = engine.play(state.board, chess.engine.Limit(0.5))
                state.apply_action(result.move)
        winner = state.get_winner()
        value = 1 if winner == player else (0 if winner == 2 else -1)
        print(player, state.board.board_fen())
        for s, p, v in zip(states, policies, [value]):
            replay_buffer.store(s, p, v)
    return replay_buffer

def create_callbacks():
    checkpoint = ModelCheckpoint(
        filepath=model_weights_path,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
        save_freq="epoch",
        verbose=1
    )
    return [checkpoint]

def train_model(model, replay_buffer: ReplayBuffer, batch_size=256, epochs=3):
    callbacks = create_callbacks()
    for epoch in range(epochs):
        states, policies, values = replay_buffer.sample(batch_size)
        states = np.squeeze(states)
        if len(states.shape) == 3:  # Eğer eksik boyut varsa
            states = np.expand_dims(states, -1)
        values = np.array(values).reshape(-1, 1)
        
        model.fit(
            states,
            {"policy_output": policies, "value_output": values},
            batch_size=batch_size,
            epochs=1,
            callbacks=callbacks,
            verbose=1
        )

def train_sigmachess(model, num_iterations=100, num_games_per_iteration=100):
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        
        mcts = MCTS(model, 1.0, 80)
        replay_buffer = self_play(mcts, num_games_per_iteration)
        train_model(model, replay_buffer)
    
    # Save the final model in Kaggle's output directory
    model.save("/kaggle/working/sigmachess_model/full_model.keras")
    model.save_weights("/kaggle/working/sigmachess_model/final_model_weights.h5")

# Create and train the model
model = create_model()

# Eğer model ağırlıkları zaten varsa yükle
if os.path.exists(model_weights_path):
    print(f"Loading weights from {model_weights_path}")
    model.load_weights(model_weights_path)

# Eğitimi başlat
train_sigmachess(model, num_iterations=7000, num_games_per_iteration=15)
