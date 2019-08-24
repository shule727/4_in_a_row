from copy import deepcopy
from mpi4py import MPI
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class MessageType():
    BOARD, REQUEST, TASK, RESULT, WAIT, STOP = 0, 1, 2, 3, 4, 5

class Game:

    Player = 0
    CPU = 1

    def __init__(self, columns = 7, rows = 50):
        self.columns = columns
        self.rows = rows
        self.board = [[None for j in range(rows)] for i in range(columns)]
        self.rowsIndex = [0 for i in range(columns)]
        self.lastPlayer = Player
        self.lastRow = 0
        self.lastColumn = 0

    def printBoard(self):
        outString = ''
        for i in range(max(self.rowsIndex)):
            line = '|'
            for j in range(self.columns):
                if self.board[j][i] == None:
                    line += ' |'
                if self.board[j][i] == Player:
                    line += 'x|'
                if self.board[j][i] == CPU:
                    line += 'o|'
            line += '\n'
            outString = line + outString
        print(outString)

    #Move is played by selecting a column in which the token is put
    def playMove(self, column, player):
        self.board[column][self.rowsIndex[column]] = player
        self.lastColumn = column
        self.lastRow = self.rowsIndex[column]
        self.rowsIndex[column] += 1
        self.lastPlayer = player

    def checkWinner(self, player = None):
        if player == None:
            player = self.lastPlayer
        if self.checkColumn(player):
            return True
        if (self.checkRow(player)):
            return True
        if (self.checkLeftDiagonal(player)):
            return True

        if(self.checkRightDiagonal(player)):
            return True

        return False

    def checkRow(self, player = None):
        if player == None:
            player = self.lastPlayer
        diff = self.lastColumn - 3
        for i in range(max(0, diff), min(self.columns-4, self.lastColumn) + 1):
            if self.board[i][self.lastRow] == player:
                if self.board[i + 1][self.lastRow] == player:
                    if self.board[i + 2][self.lastRow] == player:
                        if self.board[i + 3][self.lastRow] == player:
                            return True
        return False

    def checkColumn(self, player = None):
        if player == None:
            player = self.lastPlayer
        for i in range(self.lastRow -3, self.lastRow +1):
            if self.board[self.lastColumn][i] != player:
                return False
        return True


    def checkLeftDiagonal(self, player = None):
        if player == None:
            player = self.lastPlayer
        x, y = self.lastColumn, self.lastRow
        for i in range(4):
            if ((x < self.columns) and (y >= 0)):
                counter = 0
                for j in range(4):
                    if self.board[x - j][y + j] == player:
                        counter += 1
                if counter == 4:
                    return True
            x, y = x + 1, y -1
        return False

    def checkRightDiagonal(self, player = None):
        if player == None:
            player = self.lastPlayer
        x, y = self.lastColumn, self.lastRow
        for i in range(4):
            if ((x - 3 >= 0) and (y >= 0) ):
                counter = 0
                for j in range(4):
                    if self.board[x - j][y - j] == player:
                        counter += 1
                if counter == 4:
                    return True
            x, y = x - 1, y - 1
        return False

    #Algorithm searches the subtree of states for a given move and returns the quality of a given move.
    #State is a "win" if CPU has 4 in a row (value 1)
    #State is a "loss" if player has 4 in a row (value -1)
    #Otherwise, state is neutral and its value will be depend on the subtree states.
    #Recursive formula for value is (number_of_wins_in_depth_n - number_of_defeats_in_depth_n)/(number_of_possible_moves)
    def predict(self, depth, move = None):
        orgGame = deepcopy(self)
        if move != None:
            for m in move:
                orgGame.playMove(m, 1 - orgGame.lastPlayer)
                if orgGame.checkWinner():
                    if orgGame.lastPlayer == CPU:
                        return 1
                    else:
                        return -1


        if orgGame.checkWinner():
            if orgGame.lastPlayer == CPU:
                return 1
            else:
                return -1

        if depth == 0:
            return 0

        total = 0
        for i in range(orgGame.columns):
            game = deepcopy(orgGame)
            game.playMove(i, 1 - game.lastPlayer)
            state = self.predict(depth - 1)

            if (game.lastPlayer == Player and state == 1):
                return 1

            if (game.lastPlayer == CPU and state == -1):
                return -1

            total += state

        if total == orgGame.columns:
            return 1
        if total == -orgGame.columns:
            return -1
        return total/orgGame.columns

    #Returns the move with the best quality
    def bestMove(self, tasks):
        columns = [0 for i in range(7)]
        for k in tasks.keys():
            columns[k[0]] += tasks[k]
        maxColumn =  columns.index(max(columns))
        return maxColumn

    #Returns a task that still doesn't have calculated value
    def nextTask(self, tasks):
        for k in tasks.keys():
            if tasks.get(k) == None:
                tasks[k] = True
                return k
        return None

    #Creates dict of task with keys that represent sequence of moves
    #e.g.[0,1] represents a move in which CPU puts token in first column and player in second
    def createTasks(self):
        tasks = dict()
        for i in range(7):
            for j in range(7):
                tasks[(i,j)] = None
        return tasks




def master():
    game = Game()
    while True:
        startTime = time()
        tasks = game.createTasks()
        #Send the state of the board to the workers
        message = {'type': MessageType.BOARD, 'board': game}
        for x in range (1, size):
            comm.send(message, dest = x)
        activeWorkers = size -1

        while True:
            status = MPI.Status()
            message = comm.recv(source = MPI.ANY_SOURCE, status=status)
            source = status.Get_source()

            #If worker requests a new task, give him the one that stil isn't calculated. If there are none, tell him to wait.
            if message['type'] == MessageType.REQUEST:
                task = self.nextTask(tasks)

                if task != None:
                    message = {'type': MessageType.TASK, 'task': task}
                else:
                    activeWorkers = activeWorkers - 1
                    message = {'type': MessageType.WAIT}
                comm.send(message, dest = source)
                if activeWorkers == 0:
                    break
            elif message['type'] == MessageType.RESULT:
                tasks[message['task']] = message['result']

        nextMove = game.bestMove(tasks)
        endTime = time() - startTime
        print('Calculated in: ', endTime, 's')
        game.playMove(nextMove, 1 - game.lastPlayer)

        game.printBoard()
        print()

        if game.checkWinner():
            print('***CPU WON***')
            break

        playerMove = int(input('Play your move!'))
        game.playMove(playerMove, 1 - game.lastPlayer)

        game.printBoard()
        print()

        if game.checkWinner():
            print('***PLAYER WON***')
            break

    #Stop workers
    for i in range(1, size):
        comm.send({'type': MessageType.STOP}, dest = 1)
    return

def worker():
    game = Game()
    while True:
        message = comm.recv(source = 0)
        if message['type'] == MessageType.BOARD:
            game = message['board']
        elif message['type'] == MessageType.STOP:
            break

        while True:
            message = {'type': MessageType.REQUEST}
            comm.send(message, dest = 0)

            message = comm.recv(source=0)
            if message['type'] == MessageType.WAIT:
                break

            task = message['task']
            result = game.predict(4, task)
            message = {'type' : MessageType.RESULT, 'task': task, 'result': result}
            comm.send(message, dest = 0)

Player = 0
CPU = 1


def main():
    if rank == 0:
        master()
    else:
        worker()

if __name__ == '__main__':
    main()
