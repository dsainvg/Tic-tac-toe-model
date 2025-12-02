import torch
from board import Board

def test_win_conditions():
    """Test various win conditions for the Board class"""
    
    print("=" * 50)
    print("Testing Board Win Checking Functionality")
    print("=" * 50)
    
    # Test 1: Horizontal win - top row
    print("\nTest 1: Horizontal win - Top Row (0, 1, 2)")
    board = Board()
    board.play(1, 0)  # X at position 0
    board.play(1, 1)  # X at position 1
    result, status = board.play(1, 2)  # X at position 2
    board.printBoard()
    print(f"Result: {status}, Expected: win")
    assert status == "win", "Failed: Top row horizontal win"
    print("✓ PASSED")
    
    # Test 2: Horizontal win - middle row
    print("\nTest 2: Horizontal win - Middle Row (3, 4, 5)")
    board = Board()
    board.play(2, 3)  # O at position 3
    board.play(2, 4)  # O at position 4
    result, status = board.play(2, 5)  # O at position 5
    board.printBoard()
    print(f"Result: {status}, Expected: win")
    assert status == "win", "Failed: Middle row horizontal win"
    print("✓ PASSED")
    
    # Test 3: Horizontal win - bottom row
    print("\nTest 3: Horizontal win - Bottom Row (6, 7, 8)")
    board = Board()
    board.play(1, 6)  # X at position 6
    board.play(1, 7)  # X at position 7
    result, status = board.play(1, 8)  # X at position 8
    board.printBoard()
    print(f"Result: {status}, Expected: win")
    assert status == "win", "Failed: Bottom row horizontal win"
    print("✓ PASSED")
    
    # Test 4: Vertical win - left column
    print("\nTest 4: Vertical win - Left Column (0, 3, 6)")
    board = Board()
    board.play(1, 0)  # X at position 0
    board.play(1, 3)  # X at position 3
    result, status = board.play(1, 6)  # X at position 6
    board.printBoard()
    print(f"Result: {status}, Expected: win")
    assert status == "win", "Failed: Left column vertical win"
    print("✓ PASSED")
    
    # Test 5: Vertical win - middle column
    print("\nTest 5: Vertical win - Middle Column (1, 4, 7)")
    board = Board()
    board.play(2, 1)  # O at position 1
    board.play(2, 4)  # O at position 4
    result, status = board.play(2, 7)  # O at position 7
    board.printBoard()
    print(f"Result: {status}, Expected: win")
    assert status == "win", "Failed: Middle column vertical win"
    print("✓ PASSED")
    
    # Test 6: Vertical win - right column
    print("\nTest 6: Vertical win - Right Column (2, 5, 8)")
    board = Board()
    board.play(1, 2)  # X at position 2
    board.play(1, 5)  # X at position 5
    result, status = board.play(1, 8)  # X at position 8
    board.printBoard()
    print(f"Result: {status}, Expected: win")
    assert status == "win", "Failed: Right column vertical win"
    print("✓ PASSED")
    
    # Test 7: Diagonal win - top-left to bottom-right
    print("\nTest 7: Diagonal win - Top-Left to Bottom-Right (0, 4, 8)")
    board = Board()
    board.play(1, 0)  # X at position 0
    board.play(1, 4)  # X at position 4
    result, status = board.play(1, 8)  # X at position 8
    board.printBoard()
    print(f"Result: {status}, Expected: win")
    assert status == "win", "Failed: Diagonal (0,4,8) win"
    print("✓ PASSED")
    
    # Test 8: Diagonal win - top-right to bottom-left
    print("\nTest 8: Diagonal win - Top-Right to Bottom-Left (2, 4, 6)")
    board = Board()
    board.play(2, 2)  # O at position 2
    board.play(2, 4)  # O at position 4
    result, status = board.play(2, 6)  # O at position 6
    board.printBoard()
    print(f"Result: {status}, Expected: win")
    assert status == "win", "Failed: Diagonal (2,4,6) win"
    print("✓ PASSED")
    
    # Test 9: Draw condition
    print("\nTest 9: Draw condition (no winner, board full)")
    board = Board()
    # X O X
    # X O O
    # O X X
    board.play(1, 0)  # X
    board.play(2, 1)  # O
    board.play(1, 2)  # X
    board.play(1, 3)  # X
    board.play(2, 4)  # O
    board.play(2, 5)  # O
    board.play(2, 6)  # O
    board.play(1, 7)  # X
    result, status = board.play(1, 8)  # X
    board.printBoard()
    print(f"Result: {status}, Expected: draw")
    assert status == "draw", "Failed: Draw condition"
    print("✓ PASSED")
    
    # Test 10: Invalid move
    print("\nTest 10: Invalid move (playing on occupied position)")
    board = Board()
    board.play(1, 0)  # X at position 0
    result, status = board.play(2, 0)  # Try to play O at position 0
    board.printBoard()
    print(f"Result: {status}, Expected: invalid")
    assert status == "invalid", "Failed: Invalid move detection"
    print("✓ PASSED")
    
    # Test 11: Game in progress
    print("\nTest 11: Game in progress (no win, no draw)")
    board = Board()
    board.play(1, 0)  # X
    board.play(2, 4)  # O
    result, status = board.play(1, 8)  # X
    board.printBoard()
    print(f"Result: {status}, Expected: none")
    assert status == "none", "Failed: Game in progress"
    print("✓ PASSED")
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! ✓")
    print("=" * 50)

if __name__ == "__main__":
    test_win_conditions()
