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
    def __init__(self, model_path=None):
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
    def __init__(self, model_path=None):
        self.nn_model = ChessNN()
        if model_path:
            self.nn_model.load_state_dict(torch.load(model_path))
        self.nn_model.eval()

    def get_move(self, board):
        best_move = None
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        move_evaluations = []

        for move in board.legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            board_tensor = board_to_tensor(board_copy)
            with torch.no_grad():
                score = self.nn_model(board_tensor).item()

            move_evaluations.append((move, score))

            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        # Sort moves by score
        move_evaluations.sort(key=lambda x: x[1], reverse=(board.turn == chess.WHITE))

        reasoning = "AI's move analysis:\n"
        for i, (move, score) in enumerate(move_evaluations[:5]):  # Show top 5 moves
            reasoning += f"{i+1}. Move {move}: Evaluation = {score:.4f}\n"

        reasoning += f"\nSelected move: {best_move} with score {best_score:.4f}"

        return best_move, reasoning

# Main execution
if __name__ == "__main__":
    game = SimpleChessGame()
    game.display()