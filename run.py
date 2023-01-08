import pygame, sys, cv2, numpy as np
from pygame.locals import *
from keras.models import load_model


pygame.init()
pygame.font.init()
FONT = pygame.font.Font("freesansbold.ttf", 18)
WIDTH = 600
HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BOUNDARY = 5
SAVEIMG = False
PREDICT = True
MODEL = load_model("bestmodel.h5")
LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three",
          4: "Four", 5: "Five", 6: "Six", 7: "Seven",
          8: "Eight", 9: "Nine"}

def run_window(screen):
    wr = False
    img_cnt = 0 # 0 or 1?
    digit_xcor = []
    digit_ycor = []
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == MOUSEMOTION and wr:
                x, y = event.pos
                digit_xcor.append(x)
                digit_ycor.append(y)
                pygame.draw.circle(screen, WHITE, (x, y), 4, 0)

            if event.type == MOUSEBUTTONDOWN:
                wr = True

            if event.type == MOUSEBUTTONUP and digit_xcor and digit_ycor:
                wr = False
                digit_xcor = sorted(digit_xcor)
                digit_ycor = sorted(digit_ycor)

                min_x, max_x = max(digit_xcor[0] - BOUNDARY, 0), min(WIDTH, digit_xcor[-1] + BOUNDARY)
                min_y, max_y = max(digit_ycor[0] - BOUNDARY, 0), min(digit_ycor[-1] + BOUNDARY, HEIGHT) 

                digit_xcor = []
                digit_ycor = []

                img_arr = np.array(pygame.PixelArray(screen))[min_x : max_x, min_y : max_y].T.astype(np.float32)

                if SAVEIMG:
                    cv2.imwrite("image.png")
                    img_cnt += 1

                if PREDICT:
                    image = cv2.resize(img_arr, (28, 28)) # Modeal trained on 28x28
                    image = np.pad(image, (10, 10), 'constant', constant_values= 0)
                    image = cv2.resize(image, (28, 28)) / 255
                    label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
                    
                    textDig = FONT.render(label, True, GREEN, WHITE)
                    textRect = pygame.Surface.get_rect(textDig)
                    textRect.left, textRect.bottom = min_x, min_y
                    pygame.draw.rect(screen, GREEN, pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y), 2)
                    screen.blit(textDig, textRect)

            if event.type == KEYDOWN:
                if event.unicode == "n":
                    screen.fill(BLACK)

        pygame.display.update()

def main():
    screen =pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Number Board")
    run_window(screen)

if __name__ == "__main__":
    main()