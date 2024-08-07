import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import io
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
import random

def board_to_tensor(board):
    tensor = torch.zeros(12, 8, 8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = 0 if piece.color == chess.WHITE else 6
            piece_type = piece.piece_type - 1
            row, col = divmod(square, 8)
            tensor[color + piece_type, row, col] = 1
    return tensor

def move_to_index(move):
    return move.from_square * 64 + move.to_square

class ChessDataset(Dataset):
    def __init__(self, pgn_file, sample_size=None, max_games=None):
        self.positions = []
        self.moves = []
        
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
        
        for encoding in encodings:
            try:
                with open(pgn_file, encoding=encoding) as f:
                    game_count = 0
                    while True:
                        game = chess.pgn.read_game(f)
                        if game is None or (max_games and game_count >= max_games):
                            break
                        board = game.board()
                        for move in game.mainline_moves():
                            self.positions.append(board_to_tensor(board))
                            self.moves.append(move_to_index(move))
                            board.push(move)
                        game_count += 1
                print(f"Successfully read {game_count} games from the PGN file with {encoding} encoding.")
                break  # If successful, exit the loop
            except UnicodeDecodeError:
                print(f"Failed to decode with {encoding}, trying next encoding...")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                raise
        else:
            raise ValueError("Unable to read the PGN file with any of the attempted encodings.")

        if sample_size and sample_size < len(self.positions):
            indices = random.sample(range(len(self.positions)), sample_size)
            self.positions = [self.positions[i] for i in indices]
            self.moves = [self.moves[i] for i in indices]
            print(f"Sampled {sample_size} positions from the dataset.")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.moves[idx]

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 4096)  # 64 * 64 possible moves

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def visualize_board_evaluation(model, board):
    model.eval()
    board_tensor = board_to_tensor(board).unsqueeze(0)
    with torch.no_grad():
        move_probabilities = model(board_tensor)
    
    move_probs = move_probabilities.view(64, 64).cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(move_probs, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Move Probability')
    plt.title('Chess Board Move Evaluation')
    plt.xlabel('To Square')
    plt.ylabel('From Square')
    plt.show()

def train_model(model, train_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    example_board = chess.Board()  # An example board to visualize during training

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for positions, moves in progress_bar:
            optimizer.zero_grad()
            outputs = model(positions)
            loss = criterion(outputs, moves)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        # Visualize progress
        clear_output(wait=True)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        plot_loss(losses)
        visualize_board_evaluation(model, example_board)

    return model, losses

# Load your PGN file
pgn_file = "Kasparov.pgn"

# Create dataset and dataloader with sampling
sample_size = 10000  # Adjust this number as needed
max_games = 100  # Limit the number of games to read, adjust as needed
dataset = ChessDataset(pgn_file, sample_size=sample_size, max_games=max_games)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize and train the model
model = ChessNN()
trained_model, training_losses = train_model(model, train_loader, num_epochs=10, learning_rate=0.001)

# Final visualizations
plot_loss(training_losses)
visualize_board_evaluation(trained_model, chess.Board())

# Save the trained model
torch.save(trained_model.state_dict(), "trained_chess_model.pth")

print("Model trained and saved as 'trained_chess_model.pth'")






import random
import json
from IPython.display import display, HTML, Javascript, clear_output, SVG
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.svg
import ipywidgets as widgets

class SimpleChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.move_input = widgets.Text(description="Move:")
        self.move_button = widgets.Button(description="Make Move")
        self.move_button.on_click(self.make_move)
        self.instructions = widgets.HTML(value="<b>Instructions:</b> Enter moves in algebraic notation (e.g., 'e2e4' or 'Nf3'). Click 'Make Move' or press Enter to submit.")
        self.output = widgets.Output()
        self.layout = widgets.VBox([self.instructions, self.move_input, self.move_button, self.output])
        # self.ai = NeuralNetworkAI()
        self.ai = EnsembleAI()  # Use EnsembleAI instead of just NeuralNetworkAI


    def display_board(self):
        with self.output:
            clear_output(wait=True)
            display(SVG(chess.svg.board(board=self.board, size=600)))  # Changed size from default (400) to 300

    def make_move(self, b):
        move = self.move_input.value
        try:
            self.board.push_san(move)
            self.move_input.value = ''
            self.display_board()
            with self.output:
                print(f"Your move: {move}")
                
            # AI's turn
            ai_move, reasoning = self.ai.get_move(self.board)
            if ai_move:
                self.board.push(ai_move)
                self.display_board()
                with self.output:
                    print(reasoning)
                    print(f"AI moved: {ai_move}")
            else:
                with self.output:
                    print("AI has no valid moves")

            if self.board.is_game_over():
                with self.output:
                    print("Game over")
                    print(f"Result: {self.board.result()}")
        except ValueError:
            with self.output:
                print("Invalid move. Please try again.")

    def display(self):
        display(self.layout)
        self.display_board()

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 4096)  # 64 * 64 possible moves

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def board_to_tensor(board):
    tensor = torch.zeros(12, 8, 8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = 0 if piece.color == chess.WHITE else 6
            piece_type = piece.piece_type - 1
            row, col = divmod(square, 8)
            tensor[color + piece_type, row, col] = 1
    return tensor


class MaterialCountAI:
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }

    def evaluate_board(self, board):
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        return score

    def get_move(self, board):
        best_move = None
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        for move in board.legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            score = self.evaluate_board(board_copy)
            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score, best_move = score, move
            else:
                if score < best_score:
                    best_score, best_move = score, move
        return best_move, f"Material Count: {best_score}"

class MinimaxAI:
    def __init__(self, depth):
        self.depth = depth
        self.material_count_ai = MaterialCountAI()

    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self.material_count_ai.evaluate_board(board)

        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_move(self, board):
        best_move = None
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        for move in board.legal_moves:
            board.push(move)
            score = self.alpha_beta(board, self.depth - 1, float('-inf'), float('inf'), board.turn != chess.WHITE)
            board.pop()
            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score, best_move = score, move
            else:
                if score < best_score:
                    best_score, best_move = score, move
        return best_move, f"Minimax (depth {self.depth}): {best_score}"

class EnsembleAI:
    def __init__(self, model_path="trained_chess_model.pth"):
        self.neural_net_ai = NeuralNetworkAI(model_path)
        self.material_count_ai = MaterialCountAI()
        self.minimax_ai = MinimaxAI(depth=3)

    def get_move(self, board):
        nn_move, nn_reasoning = self.neural_net_ai.get_move(board)
        mc_move, mc_reasoning = self.material_count_ai.get_move(board)
        mm_move, mm_reasoning = self.minimax_ai.get_move(board)

        # Simple voting system
        moves = [nn_move, mc_move, mm_move]
        selected_move = max(set(moves), key=moves.count)

        reasoning = f"Neural Network: {nn_reasoning}\n"
        reasoning += f"Material Count: {mc_reasoning}\n"
        reasoning += f"Minimax: {mm_reasoning}\n"
        reasoning += f"Selected move: {selected_move} (by majority vote)"

        return selected_move, reasoning


class NeuralNetworkAI:
    def __init__(self, model_path="trained_chess_model.pth"):
        self.nn_model = ChessNN()
        self.nn_model.load_state_dict(torch.load(model_path))
        self.nn_model.eval()

    def get_move(self, board):
        board_tensor = board_to_tensor(board).unsqueeze(0)
        
        with torch.no_grad():
            move_probabilities = self.nn_model(board_tensor)
        
        move_index = torch.argmax(move_probabilities).item()
        from_square = move_index // 64
        to_square = move_index % 64
        move = chess.Move(from_square, to_square)
        
        if move in board.legal_moves:
            return move, f"Neural Network evaluation: {move_probabilities.max().item():.4f}"
        else:
            # Fallback to a random legal move if the NN suggests an illegal move
            return random.choice(list(board.legal_moves)), "Fallback to random move"

# Main execution
if __name__ == "__main__":
    game = SimpleChessGame()
    game.display()