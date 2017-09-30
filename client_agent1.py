import random
import numpy as np
import pickle
from socket import *
import sys

#Server variables
serverName='10.10.187.175'
serverPort=12000

# Setting up connection to server
clientSocket = socket(AF_INET, SOCK_STREAM)  # define and open client socket on client (with IPV4 and TCP)
clientSocket.connect((serverName, serverPort))  # connecting to server

#Boards Parameters
BOARD_WIDTH=50
BOARD_HEIGHT = 50


#def drawMap(mySnake, enemySnake):
#	allSnakes = enemySnake.append(mySnake)
#	for snake in allSnakes[:-1]:
#		if snake[0]!="dead":
#			if (snake[0]=="invisible")&&():
#
#			else:
#	else:
# enemyBrush = 1
# mySnakeHeadBrush = 3
# normalAppleBrush = 5
# superAppleBrush = 6

brushSuperApple = np.float32(6)
brushNormalApple = np.float32(5)
brushEnemyHead = np.float32(2)
brushEnemyBody = np.float32(1)
brushWinnerHead = np.float32(4)
brushWinnerBody = brushEnemyBody

def addSnakeToMap(snakeCoordsToAdd, gameMap):
	for coords in snakeCoordsToAdd:
		gameMap[coords[0]][coords[1]] = brushEnemyBody
	return gameMap

def sphere (winningSnake, board):
	if winningSnake[0]== "invisible":
		isInvisible=True
	else:
		isInvisible = False

	# head = tuple(map(int, winningSnake[3 + 2*isInvisible].split(',')))
	head = winningSnake[3 + 2 * isInvisible]
	newHead = (head[1],BOARD_HEIGHT - 1 - head[0])

	if (newHead[1]<(BOARD_HEIGHT+1)/2):
		board = np.roll(board,(newHead[1]+2+(BOARD_HEIGHT+1)/2),axis=0)
	elif (newHead[1]>(BOARD_HEIGHT+1)/2):
		board = np.roll(board, (newHead[1]-(BOARD_HEIGHT+1)/2), axis=0)

	if (newHead[0]<(BOARD_WIDTH+1)/2):
		board = np.roll(board,((BOARD_WIDTH+1)/2-newHead[0]),axis=1)
	elif (newHead[0]>(BOARD_WIDTH+1)/2):
		board = np.roll(board, (BOARD_WIDTH-newHead[0]+2+(BOARD_WIDTH+1)/2), axis=1)

	# firstKink = tuple(map(int, winningSnake[7 + 2*isInvisible].split(',')))
	firstKink = winningSnake[4 + 2 * isInvisible]
	newfirstKink = (firstKink[1], BOARD_HEIGHT - 1 - firstKink[0])
	if (newHead[0] < newfirstKink[0]): #dir=left
		board = np.rot90(board,3)
		board = np.roll(board, -1, axis=1)
		board = np.roll(board, -3, axis=0)
	elif (newHead[0] > newfirstKink[0]): #dir=right
		board =  np.rot90(board, 1)
		board = np.roll(board, -2, axis=0)
	elif (newHead[1] < newfirstKink [1]): #dir=down
		board = np.rot90(board, 2)
		board = np.roll(board, -3, axis=0)
	else:                               #dir=up
		board = np.roll(board, -1, axis=1)
		board = np.roll(board, -2, axis=0)

	return board

if __name__ == "__main__":
#def playGame():
	line = raw_input()
	split = line.split(" ")
	numSnakes = int(split[0])

	while(True):
		gameMap = np.zeros((50, 50),
						   dtype=np.float32)  # [[0 for i in xrange(50)] for j in xrange(50)] #gameMap is a 50x50 zeroes
		allSnakes = []

		line = raw_input()
		if "Game Over" in line:
			break
		superApple = tuple(map(int, line.split(' ')))
		gameMap[superApple[0]][superApple[1]] = brushSuperApple
		normalApple = tuple(map(int, raw_input().split(' ')))
		gameMap[normalApple[0]][normalApple[1]] = brushNormalApple
		# gameMap[mySnake[3+(mySnake[0]=="invisible")][0]][mySnake[3+(mySnake[0]=="invisible")][1]] = mySnakeHeadBrush
		mySnakeIndex = int(raw_input())
		enemySnake = []
		for i in range(numSnakes):
			line = raw_input()
			if(i == mySnakeIndex):
				mySnake=line.split(' ')
				# mySnakeCoords = []
				for count,coords in enumerate(mySnake[3+(mySnake[0]=="invisible"):]):
					mySnake[3 + (mySnake[0] == "invisible")+count] = tuple(map(int, coords.split(',')))
					# mySnakeCoords.append(tuple(map(int, coords.split(','))))
				if(mySnake[0] != "dead"):
					#addSnakeToMap(mySnakeCoords)
					gameMap = addSnakeToMap(mySnake[3+(mySnake[0]=="invisible"):], gameMap)
				gameMap[mySnake[3+(mySnake[0]=="invisible")][0]][mySnake[3+(mySnake[0]=="invisible")][1]] = brushWinnerHead #overwriting our head to world map
			else:
				enemySnake.append(line.split(' '))
				#enemySnakeCoords = []
				for count,coords in enumerate(enemySnake[-1][3+(enemySnake[-1][0]=="invisible"):]):
					enemySnake[-1][3 + (enemySnake[-1][0] == "invisible")+count] = tuple(map(int, coords.split(',')))
					#enemySnakeCoords.append(tuple(map(int, coords.split(','))))
				if(enemySnake[-1][0] != "dead" and enemySnake[-1][0] != "invisible"):
					#addSnakeToMap(enemySnakeCoords)
					gameMap = addSnakeToMap(enemySnake[-1][3+(enemySnake[-1][0]=="invisible"):],gameMap)
				gameMap[enemySnake[-1][3 + (enemySnake[-1][0] == "invisible")][0]][enemySnake[-1][3 + (enemySnake[-1][0] == "invisible")][1]] = brushEnemyHead

		#pad here
		gameMap = np.pad(gameMap, 1, 'constant', constant_values=(brushEnemyBody))
		#sphere here
		spheredGameMap = sphere(mySnake, gameMap)
		spheredGameMap = spheredGameMap.flatten()
		# gameMapToSend = pickle.dump(spheredGameMap)
		# print(spheredGameMap.shape)
		pickledSpheredGameMap = pickle.dumps(spheredGameMap)
		clientSocket.send(str(sys.getsizeof(pickledSpheredGameMap)))
		clientSocket.send(pickledSpheredGameMap)

		# print("Waiting for move from server...")
		replyAnswerSerialized = clientSocket.recv(32)
		# print("Received for move from server")
		# replyAnswer = pickle.loads(replyAnswerSerialized)
		# print(replyAnswer)
		print(replyAnswerSerialized)

	# print ("Closing Connection...")
	clientSocket.close()
	# print("Connection closed")



     #    return spheredGameMap
	# return

