from classes import GameState, Node, MCTS
from models import sigmachess_network

import numpy as np
import chess
import socket

# Model oluşturuluyor
model = sigmachess_network()

# Kaggle üzerinde model ağırlıklarının yolu
model_weights_path = "/kaggle/working/checkpoints/model_checkpoint.weights.h5"

# Model ağırlıklarını yükleme
model.load_weights(model_weights_path)

gamestate = GameState()
mcts = MCTS(model, simulations=10)

# Sunucu yapılandırması
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("0.0.0.0", 5051))  # Kaggle ortamında yerel ağ adresi

def start_server():
    s.listen()

    print("Sunucu başlatıldı ve bağlantı bekleniyor...")
    client_socket, client_address = s.accept()
    print(f"Bağlantı kabul edildi: {client_address}")

    while client_socket:
        data = client_socket.recv(1024)
        move = data.decode()

        print(f"Alınan hamle: {move}")

        if move == "exit":
            print("Bağlantı sonlandırılıyor...")
            s.close()
            break
        
        # Hamle oyuna uygulanıyor
        gamestate.apply_action(chess.Move.from_uci(move))

        # MCTS ile en iyi hamle seçiliyor
        best_move = mcts.run(gamestate)
        action = np.random.choice(len(best_move), p=best_move)
        m, state = gamestate.get_next_state(action)

        # En iyi hamle gönderiliyor
        client_socket.sendall(chess.Move.uci(m).encode())

try:
    start_server()
except KeyboardInterrupt:
    s.close()
finally:
    s.close()
