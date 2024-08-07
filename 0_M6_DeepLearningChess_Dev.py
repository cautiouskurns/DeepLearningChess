import random
import json
from IPython.display import display, HTML, Javascript, clear_output
import torch
import torch.nn as nn
import torch.optim as optim
import chess

# Remove these classes as they're no longer needed
# class PieceType(Enum):
# class Color(Enum):
# class Piece:

class ChessGame:
    def __init__(self):
        self.board = chess.Board()

    def display(self):
        print(self.board)

    def make_move(self, move):
        try:
            self.board.push_san(move)
            return True
        except ValueError:
            return False

    def is_game_over(self):
        return self.board.is_game_over()

    def display_js_board(self):
        """Display the chessboard using chessboard.js"""
        js_code = f"""
        var board = Chessboard('board', '{self.board.fen()}');
        """
        display(Javascript(js_code))

class RandomAI:
    def get_move(self, board):
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None

class MaterialCountAI:
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
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
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move

class MinimaxAI:
    def __init__(self, depth):
        self.depth = depth
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100
        }

    def evaluate_board(self, board):
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        return score

    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

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

        print("AI is considering the following moves:")
        for move in board.legal_moves:
            board.push(move)
            score = self.alpha_beta(board, self.depth - 1, float('-inf'), float('inf'), board.turn != chess.WHITE)
            board.pop()

            print(f"Move {move}: Evaluated score = {score}")

            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        if best_move:
            print(f"AI selected move: {best_move} with score {best_score}")
        else:
            print("AI could not find a valid move")

        return best_move

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.fc1 = nn.Linear(64 * 12, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

def board_to_tensor(board):
    tensor = torch.zeros(1, 12, 8, 8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = 0 if piece.color == chess.WHITE else 6
            piece_type = piece.piece_type - 1
            row, col = divmod(square, 8)
            tensor[0, color + piece_type, row, col] = 1
    return tensor.view(1, -1)

class NeuralNetworkAI:
    def __init__(self, model_path=None):
        self.nn_model = ChessNN()
        if model_path:
            self.nn_model.load_state_dict(torch.load(model_path))
        self.nn_model.eval()

    def get_move(self, board):
        best_move = None
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')

        for move in board.legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            board_tensor = board_to_tensor(board_copy)
            with torch.no_grad():
                score = self.nn_model(board_tensor).item()

            print(f"Move {move}: NN Evaluation = {score}")

            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        if best_move:
            print(f"AI selected move: {best_move} with score {best_score}")
        else:
            print("AI could not find a valid move")

        return best_move

def play_game_with_ai():
    game = ChessGame()
    ai = NeuralNetworkAI()  # You can change this to any other AI class

    while not game.is_game_over():
        game.display()
        if game.board.turn == chess.WHITE:
            move = input("Enter your move (e.g., e2e4): ")
            if game.make_move(move):
                print("Move successful")
            else:
                print("Invalid move")
                continue
        else:
            print("AI's turn (Black)")
            ai_move = ai.get_move(game.board)
            if ai_move:
                game.board.push(ai_move)
                print(f"AI moved: {ai_move}")
            else:
                print("AI has no valid moves")
                break

    print("Game over")
    print(f"Result: {game.board.result()}")

if __name__ == "__main__":
    play_game_with_ai()