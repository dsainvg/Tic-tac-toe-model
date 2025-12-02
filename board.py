import torch

class Board:
    def __init__(self):
        self.board = torch.zeros(9, dtype=torch.float32)
        self.sum = 0
        self.game = []
    
    def printBoard(self):
        for i in range(3):
            print(self.board[i*3:(i+1)*3])

    def _play(self, id, val):
        if self.board[val].item() == 0:
            self.sum += 1
            self.board[val] = id
            self.game.append((id, val))
            return True
        return False
    
    def _checkWin(self, id, val):
        b = self.board  # shorthand for readability
        match val:
            case 0:
                if (b[0] == b[1]).item() and (b[1] == b[2]).item() and (b[0] == id).item():
                    return True
                if (b[0] == b[3]).item() and (b[3] == b[6]).item() and (b[0] == id).item():
                    return True
                if (b[0] == b[4]).item() and (b[4] == b[8]).item() and (b[0] == id).item():
                    return True
            case 1:
                if (b[0] == b[1]).item() and (b[1] == b[2]).item() and (b[0] == id).item():
                    return True
                if (b[1] == b[4]).item() and (b[4] == b[7]).item() and (b[1] == id).item():
                    return True
            case 2:
                if (b[0] == b[1]).item() and (b[1] == b[2]).item() and (b[0] == id).item():
                    return True
                if (b[2] == b[5]).item() and (b[5] == b[8]).item() and (b[2] == id).item():
                    return True
                if (b[2] == b[4]).item() and (b[4] == b[6]).item() and (b[2] == id).item():
                    return True
            case 3:
                if (b[3] == b[4]).item() and (b[4] == b[5]).item() and (b[3] == id).item():
                    return True
                if (b[0] == b[3]).item() and (b[3] == b[6]).item() and (b[0] == id).item():
                    return True
            case 4:
                if (b[3] == b[4]).item() and (b[4] == b[5]).item() and (b[3] == id).item():
                    return True
                if (b[1] == b[4]).item() and (b[4] == b[7]).item() and (b[1] == id).item():
                    return True
                if (b[0] == b[4]).item() and (b[4] == b[8]).item() and (b[0] == id).item():
                    return True
                if (b[2] == b[4]).item() and (b[4] == b[6]).item() and (b[2] == id).item():
                    return True
            case 5:
                if (b[3] == b[4]).item() and (b[4] == b[5]).item() and (b[3] == id).item():
                    return True
                if (b[2] == b[5]).item() and (b[5] == b[8]).item() and (b[2] == id).item():
                    return True
            case 6:
                if (b[6] == b[7]).item() and (b[7] == b[8]).item() and (b[6] == id).item():
                    return True
                if (b[0] == b[3]).item() and (b[3] == b[6]).item() and (b[0] == id).item():
                    return True
                if (b[2] == b[4]).item() and (b[4] == b[6]).item() and (b[2] == id).item():
                    return True
            case 7:
                if (b[6] == b[7]).item() and (b[7] == b[8]).item() and (b[6] == id).item():
                    return True
                if (b[1] == b[4]).item() and (b[4] == b[7]).item() and (b[1] == id).item():
                    return True
            case 8:
                if (b[6] == b[7]).item() and (b[7] == b[8]).item() and (b[6] == id).item():
                    return True
                if (b[2] == b[5]).item() and (b[5] == b[8]).item() and (b[2] == id).item():
                    return True
                if (b[0] == b[4]).item() and (b[4] == b[8]).item() and (b[0] == id).item():
                    return True            
        return False
    
    def play(self, id, val):
        if self._play(id, val):
            if self._checkWin(id,val):
                return True,"win"
            if self.sum == 9:
                return True,"draw"
            return False,"none"
        return True,"invalid"
    
    def clear(self):
        self.board = torch.zeros(9, dtype=torch.float32)
        self.sum = 0
        self.game = []
    
    @property
    def board_state(self):
        return self.board
    
    @property
    def game_state(self):
        return self.game
