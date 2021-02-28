#======================================================================
# Github: https://github.com/thjsimmons
#======================================================================

import os,sys
import pygame 
import cv2
import numpy as np
from utils import *

def dist2(A, B): # 2D Euclidean vector distance formula:
    return ((A[0]-B[0])**2.0 + (A[1]-B[1])**2.0)**0.5

class Image: # Board tiled with image objects
    def __init__(self, entry, position):
        self.name = entry[0]
        self.IMG = entry[1]
        self.cv2_IMG = entry[2]
        self.rect = pygame.Rect(position[1], position[0], self.cv2_IMG.shape[0], self.cv2_IMG.shape[1]) # image rect is just image size 

    def set_pos(self, position):
        self.rect = pygame.Rect(position[1], position[0], self.cv2_IMG.shape[0], self.cv2_IMG.shape[1])

class Board:
    def __init__(self, screen, board_size, image_size, SAVED):
        # 
        self.screen = screen
        self.board_size = board_size
        self.image_size = image_size 
        self.images = []
        self.names = fromFolder("resized/")
        self.entries = []
        self.MOVING = False

        # Get Images, Names from Folder
        count = 0
        for i in range(len(SAVED)):
            name = SAVED[i][1]
            if count < board_size[0]*board_size[1]+1:
                img = pygame.image.load(os.path.join("resized/" + name)) # pygame.surface object 
                cv2_img = cv2.imread("resized/" + name)
                if not name == "null.jpg":
                    self.entries.append([name, img, cv2_img])
                count +=1

        null_cv2_img = cv2.imread("resized/null.jpg")
        null_img = pygame.image.load(os.path.join("resized/null.jpg")) 
        null_entry = ["null.jpg", null_img, null_cv2_img]
        self.null_image = Image(null_entry, (0,0))
        #self.moving_image = null_entry
   
        for i in range(len(self.entries)):
            entry = self.entries[i]
            img = entry[1]
            col = (i % self.board_size[0]) 
            row = int(np.floor(i * 1.0/self.board_size[0]))
            x = col * self.image_size[0]
            y = row * self.image_size[1]
            self.images.append(Image(self.entries[i], (x,y))) # (self, name, img, position, index):
            self.screen.blit(img, (x,y))
        pygame.display.flip()

        self.index_clicked = -1
        self.null_index = -1
        
    def index2center(self, i):
        col = (i % self.board_size[0]) 
        row = int(np.floor(i * 1.0/self.board_size[1]))
        # top left corner
        x = col * self.size[0] 
        y = row * self.size[1]
        x_center = x + self.image_size[0]/2.0
        y_center = y + self.image_size[0]/2.0
        return (x_center, y_center)

    def index2corner(self, i):
        col = (i % self.board_size[0]) 
        row = int(np.floor(i * 1.0/self.board_size[0]))
        x = col * self.image_size[0]
        y = row * self.image_size[1]
        return (y, x)

    def getNearest(self, position):
        mindex = -1
        val = 9999
        for i in range(len(self.images)):
            d = dist2(position, self.images[i].rect.center)

            if mindex == -1 or d < val:
                mindex = i 
                val = d 
        
        return mindex

    def shift(self): 
        # keep popping and inserting null_image at mindex, re-rendering images
        for i in range(len(self.images)):
            self.images[i].set_pos(self.index2corner(i))
            self.screen.blit(self.images[i].IMG, self.images[i].rect)

        self.moving_image.rect.center = pygame.mouse.get_pos() # moving image tracks mouse position
        mindex = self.getNearest(pygame.mouse.get_pos()) # get nearest image to mouse
        self.images.insert(mindex, self.images.pop(self.null_index))
        self.null_index = mindex
        
class Game:
    def __init__(self):
        Nx = 10
        Ny = 10
        W = 90
        H = 90
        self.board = Board(pygame.display.set_mode((Nx*W, Ny*H)), (Nx, Ny), (W, H), self.Load())
    
    def Load(self):
        file = open("sorted/save.txt")
        entries = [] 

        while True:
            line = file.readline()
            if not line:
                break
            entries.append(line.replace(" ", "").replace("\n", "").split(","))
        file.close()
        return entries

    def Save(self, BOARD):
        lines = []

        for i in range(len(BOARD.images)):
            lines.append(str(i) + ", " + BOARD.images[i].name)
        with open('sorted/save.txt', 'w') as file:
            for line in lines:
                print >> file, line
        return 0

    def Write(self, BOARD): # on w key write folder
        print("Writing mosaic")
        h = BOARD.image_size[1]
        w = BOARD.image_size[0]
        Nx = BOARD.board_size[0]
        Ny = BOARD.board_size[1]

        M = newImg(Nx * w, Ny * h, [0,0,0])

        for i in range(92): # Obtain 92 via reading folder
            x = (i % Ny) * w
            y = int(np.floor(i*1.0/Ny)*h) 
            image = BOARD.images[i]
            M[y:y+h, x:x+w] = image.cv2_IMG
        
        cv2.imwrite("sorted/mosaic.jpg", scale_by(1,M))

    def getAllNames(self, BOARD):
        output = []
        for image in BOARD.images:
            output.append(image.name)
        return output

    def Main(self):
        BOARD = self.board
        iterations = 0

        while True: # render game until user quites # main loop 
            if iterations == 100:
                iterations = 0
                BOARD.screen.fill((0,0,0))

            iterations += 1
            for event in pygame.event.get():
                # Mouse Events: 
                if event.type == pygame.MOUSEBUTTONDOWN: # mouse button hold down ?
                    for index in range(len(BOARD.images)):

                        if BOARD.images[index].rect.collidepoint(event.pos):
                            BOARD.moving_image =  BOARD.images.pop(index) # pop moving_image, will now follow mouse 
                            BOARD.null_index = index
                            BOARD.images.insert(BOARD.null_index,  BOARD.null_image)
                            BOARD.null_image.set_pos(BOARD.index2corner(BOARD.null_index))
                            BOARD.MOVING = True

                elif event.type == pygame.MOUSEBUTTONUP: # Release -> Drop image where null_image is rendered (mindex)
                    if BOARD.MOVING:
                        BOARD.MOVING = False
                        BOARD.images.pop(BOARD.null_index)
                        BOARD.images.insert(BOARD.null_index,  BOARD.moving_image) 
                        BOARD.moving_image.set_pos(BOARD.index2corner(BOARD.null_index))
                        BOARD.null_index = None
                        
                        names = self.getAllNames(BOARD)
                        BOARD.screen.fill((0,0,0))
                        for i in range(len(BOARD.images)):
                            BOARD.images[i].set_pos(BOARD.index2corner(i))
                            BOARD.screen.blit(BOARD.images[i].IMG, BOARD.images[i].rect)

                        pygame.display.flip()
                        
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        BOARD.screen.fill((0,0,0))
                        for i in range(len(BOARD.images)):
                            BOARD.screen.blit(BOARD.images[i].IMG, BOARD.images[i].rect)
                        self.Save(BOARD)
                        self.Write(BOARD)
       
                # Exit Event 
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            if BOARD.MOVING:
                BOARD.shift()
                BOARD.screen.blit(BOARD.moving_image.IMG, BOARD.moving_image.rect)
                pygame.display.flip()
            
new_game = Game()
new_game.Main()
