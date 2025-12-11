import pickle
import torch
import torch.nn as nn
from board import Board

# Model architecture (must match training.ipynb)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(2*input_size, hidden_sizes[0])
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.l4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.l5 = nn.Linear(hidden_sizes[3], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        xin = torch.stack((self.relu(x),self.relu(-x)), dim=1).flatten()
        logits = torch.zeros_like(x)  # Create a tensor of all 0s
        logits = logits.masked_fill(x != 0, float('-inf'))
        out = self.l1(xin)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        out = out + logits
        out = self.softmax(out)
        return out

def load_model(model_path='models/model.pth'):
    """Load the trained model from .pth (state dict) or .pkl file
    
    Args:
        model_path: Path to the model file (.pth or .pkl)
    
    Returns:
        Loaded model in evaluation mode
    """
    print(f"Loading model from {model_path}...")
    
    # Model hyperparameters (must match training.ipynb)
    input_size = 9
    hidden_sizes = (256, 512, 64, 32)
    num_classes = 9
    
    if model_path.endswith('.pth'):
        # Load from PyTorch state dict (recommended method)
        model = NeuralNet(input_size, hidden_sizes, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print("Model loaded from .pth file (state dict)")
    elif model_path.endswith('.pkl'):
        # Load from pickle file (legacy method)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded from .pkl file (pickled object)")
    else:
        raise ValueError(f"Unsupported file format: {model_path}. Use .pth or .pkl")
    
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!\n")
    return model

def print_board_nice(board):
    """Print board in a nice format"""
    print("\nCurrent Board State:")
    print("  0 | 1 | 2")
    print("  ---------")
    print("  3 | 4 | 5")
    print("  ---------")
    print("  6 | 7 | 8")
    print()
    
    state = board.board_state
    for i in range(3):
        row = []
        for j in range(3):
            idx = i * 3 + j
            val = state[idx].item()
            if val == 0:
                row.append(str(idx))
            elif val == 1:
                row.append('X')
            else:
                row.append('O')
        print(f"  {row[0]} | {row[1]} | {row[2]}")
        if i < 2:
            print("  ---------")
    print()

def get_model_prediction(model, board):
    """Get model's prediction and show probabilities"""
    with torch.no_grad():
        board_tensor = board.board_state.unsqueeze(0)
        prediction = model(board_tensor)
        
        print("\nModel's Move Probabilities:")
        for i in range(9):
            status = "OCCUPIED" if board.board_state[i] != 0 else "FREE"
            print(f"Position {i}: {prediction[0][i].item():.4f} ({status})")
        
        # Get best move
        valid_positions = (board.board_state == 0).nonzero().flatten()
        if len(valid_positions) > 0:
            probs = prediction[0].clone()
            for i in range(9):
                if board.board_state[i] != 0:
                    probs[i] = -1
            best_move = probs.argmax().item()
            print(f"\nAI chooses position: {best_move}")
            return best_move
        return None

def _print_prediction_for_board(label, model, board):
    print(f"\n{label}:")
    with torch.no_grad():
        prediction = model(board.board_state.unsqueeze(0))[0]
    for i, prob in enumerate(prediction):
        status = "OCCUPIED" if board.board_state[i] != 0 else "FREE"
        print(f"Position {i}: {prob.item():.6f} ({status})")
    
    # Check for symmetry issues
    free_positions = [i for i in range(9) if board.board_state[i] == 0]
    if len(free_positions) > 1:
        probs = [prediction[i].item() for i in free_positions]
        max_prob = max(probs)
        min_prob = min(probs)
        if max_prob > 0 and min_prob / max_prob < 0.01:  # More than 100x difference
            print(f"  ⚠️  Symmetry issue detected: max={max_prob:.6f}, min={min_prob:.6f}")

def check_prediction_probabilities(model, opponent_id=-1):
    """Check prediction probabilities for an empty board and each single opponent placement"""
    empty_board = Board()
    _print_prediction_for_board("Empty board", model, empty_board)

    for pos in range(9):
        board = Board()
        board.play(opponent_id, pos)
        _print_prediction_for_board(f"Opponent only at {pos}", model, board)

def play_game_human_vs_ai(model):
    """Play a game: human (X=1) vs AI (O=-1)"""
    board = Board()
    print("\n" + "="*50)
    print("New Game: You (X) vs AI (O)")
    print("="*50)
    print_board_nice(board)
    
    current_player = 1  # Human starts (X)
    
    while True:
        if current_player == 1:
            # Human's turn
            print("\nYour turn (X)!")
            try:
                move = int(input("Enter position (0-8): "))
                if move < 0 or move > 8:
                    print("Invalid position! Choose 0-8.")
                    continue
            except ValueError:
                print("Invalid input! Enter a number.")
                continue
            
            status, result = board.play(1, move)
            if result == "invalid":
                print("Invalid move! Position already occupied.")
                continue
                
        else:
            # AI's turn
            print("\nAI's turn (O)...")
            move = get_model_prediction(model, board)
            if move is None:
                print("No valid moves!")
                break
            status, result = board.play(-1, move)
        
        print_board_nice(board)
        
        if result == "win":
            winner = "You" if current_player == 1 else "AI"
            print(f"\n🎉 {winner} won!")
            break
        elif result == "draw":
            print("\n🤝 It's a draw!")
            break
        
        # Switch player
        current_player = -current_player

def ai_vs_ai(model, num_games=5, verbose=True):
    """Watch AI play against itself"""
    print(f"\n{'='*50}")
    print(f"AI vs AI - {num_games} games")
    print(f"{'='*50}\n")
    
    wins = 0
    draws = 0
    
    for game_num in range(num_games):
        board = Board()
        print(f"\n{'='*50}")
        print(f"Game {game_num + 1}:")
        print(f"{'='*50}")
        current_player = 1 if game_num % 2 == 0 else -1
        
        # Show initial empty board
        print_board_nice(board)
        
        for turn in range(9):
            print(f"\nTurn {turn + 1} - Player {'X' if current_player == 1 else 'O'}'s move:")
            
            # Get model's move
            with torch.no_grad():
                board_tensor = board.board_state.unsqueeze(0)
                prediction = model(board_tensor)
                
                # Get best valid move
                valid_positions = (board.board_state == 0).nonzero().flatten()
                if len(valid_positions) == 0:
                    break
                    
                probs = prediction[0].clone()
                for i in range(9):
                    if board.board_state[i] != 0:
                        probs[i] = -1
                move = probs.argmax().item()
            
            print(f"AI chooses position: {move}")
            status, result = board.play(current_player, move)
            
            # Show board after each move
            print_board_nice(board)
            
            if result == "win":
                print(f"🎉 Player {'X' if current_player == 1 else 'O'} won!")
                if current_player == 1:
                    wins += 1
                break
            elif result == "draw":
                print("🤝 It's a draw!")
                draws += 1
                break
            
            current_player = -current_player
    
    print(f"\n{'='*50}")
    print(f"Final Results:")
    print(f"X wins: {wins}")
    print(f"O wins: {num_games - wins - draws}")
    print(f"Draws: {draws}")
    print(f"{'='*50}\n")

def test_specific_board(model):
    """Test model on a specific board state"""
    board = Board()
    print("\nSet up a board state manually.")
    print("Enter moves in format: player position (e.g., '1 0' for X at position 0)")
    print("Enter 'done' when finished.\n")
    
    while True:
        cmd = input("Move (player position) or 'done': ").strip()
        if cmd.lower() == 'done':
            break
        
        try:
            parts = cmd.split()
            player = int(parts[0])
            pos = int(parts[1])
            if player not in [1, -1]:
                print("Player must be 1 (X) or -1 (O)")
                continue
            if pos < 0 or pos > 8:
                print("Position must be 0-8")
                continue
            
            status, result = board.play(player, pos)
            if result == "invalid":
                print("Invalid move!")
            else:
                print_board_nice(board)
        except (ValueError, IndexError):
            print("Invalid input! Use format: player position (e.g., '1 0')")
    
    print_board_nice(board)
    get_model_prediction(model, board)

def main(model_path=None):
    """Check prediction probabilities for empty board and single opponent placements
    
    Args:
        model_path: Path to model file. If None, tries model.pth first, then model.pkl
    """
    # Try to load model
    if model_path is None:
        # Try .pth first (recommended), fall back to .pkl
        import os
        if os.path.exists('model.pth'):
            model_path = 'model.pth'
        elif os.path.exists('models/model.pth'):
            model_path = 'models/model.pth'
        elif os.path.exists('models/model.pkl'):
            model_path = 'models/model.pkl'
            print("Note: Using legacy .pkl format. Consider saving as .pth instead.")
        else:
            print("Error: No model file found!")
            print("Please ensure either model.pth, models/model.pth or models/model.pkl exists.")
            return
    
    model = load_model(model_path)
    
    print("\n" + "="*50)
    print("Checking Prediction Probabilities")
    print("="*50)
    check_prediction_probabilities(model, opponent_id=-1)

if __name__ == "__main__":
    main()
