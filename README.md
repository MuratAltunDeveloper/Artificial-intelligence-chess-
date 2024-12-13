# Artificial intelligence chess 
 The code I developed is an artificial intelligence algorithm that plays chess by combining MCTS and deep learning. GameState translates the state on the board into tensor format. MCTS chooses strategy by simulating moves, while the TensorFlow model predicts move probabilities and win expectation. ReplayBuffer stores game data for learning.

 Our prepare_validation_data() function creates validation data. We also did a fair comparison of the performance of different models with the same validation data set in deep_learning_train_model_Google Colab.
 The trained version of the model in Kaggle is the Kaggle_trainmodel.ipynb file.
 NOTES!!:
 Check the trained model output from "https://lichess.org/" and also download the latest version of Stockfish and name it as " \Sigma-Chess-main\sigmachess\stockfish"
 
![WhatsApp Görsel 2024-12-08 saat 00 22 45_c6b5f140](https://github.com/user-attachments/assets/fdf016a9-a46b-41f2-b93c-b8bf0d588f0a)
