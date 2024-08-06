from enum import Enum, auto

class PieceType(Enum):
    PAWN = auto()
    KNIGHT = auto()
    BISHOP = auto()
    ROOK = auto()
    QUEEN = auto()
    KING = auto()

class Color(Enum):
    WHITE = auto()
    BLACK = auto()

class Piece:
    def __init__(self, piece_type, color):
        self.piece_type = piece_type
        self.color = color

class ChessBoard:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = Color.WHITE
        self.initialize_board()

    def initialize_board(self):
        # Set up pawns
        for col in range(8):
            self.board[1][col] = Piece(PieceType.PAWN, Color.WHITE)
            self.board[6][col] = Piece(PieceType.PAWN, Color.BLACK)

        # Set up other pieces
        piece_order = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
                       PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]
        
        for col, piece_type in enumerate(piece_order):
            self.board[0][col] = Piece(piece_type, Color.WHITE)
            self.board[7][col] = Piece(piece_type, Color.BLACK)

    def display(self):
        piece_symbols = {
            (PieceType.PAWN, Color.WHITE): '♙', (PieceType.PAWN, Color.BLACK): '♟',
            (PieceType.KNIGHT, Color.WHITE): '♘', (PieceType.KNIGHT, Color.BLACK): '♞',
            (PieceType.BISHOP, Color.WHITE): '♗', (PieceType.BISHOP, Color.BLACK): '♝',
            (PieceType.ROOK, Color.WHITE): '♖', (PieceType.ROOK, Color.BLACK): '♜',
            (PieceType.QUEEN, Color.WHITE): '♕', (PieceType.QUEEN, Color.BLACK): '♛',
            (PieceType.KING, Color.WHITE): '♔', (PieceType.KING, Color.BLACK): '♚'
        }

        print("   a  b  c  d  e  f  g  h")  # Column labels
        print(" +--+--+--+--+--+--+--+--+")  # Top border

        for row in range(7, -1, -1):
            print(f"{row + 1}|", end="")  # Row label and left border
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    print(f"{piece_symbols[(piece.piece_type, piece.color)]}", end="|")
                else:
                    print("  ", end="|")  # Empty square
            print(f"{row + 1}")  # Right border and row label
            print(" +--+--+--+--+--+--+--+--+")  # Horizontal separator

        print("   a  b  c  d  e  f  g  h")  # Column labels

if __name__ == "__main__":
    board = ChessBoard()
    board.display()