import pygame
import cv2
import numpy as np
import torch
import pickle

largura = 400
altura = 400
frames = 120
cor_pincel = (255, 255, 255)
cor_branca = (0, 0, 0)

args = {
    'batch_size': 1000,
    'num_workers': 4,
    'num_classes': 10,
    'lr': 1e-4,
    'weight_decay': 5e-4,
    'num_epochs': 120
}

if torch.cuda.is_available():
  args['device'] = torch.device('cuda')
else:
  args['device'] = torch.device('cpu')


def main(frames):
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont(None, 50)
    tela = pygame.display.set_mode([largura, altura])
    tela.fill(cor_branca)
    relogio = pygame.time.Clock()
    sair = False
    while sair is False:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                pintar = True
                while pintar:
                    pygame.display.update()
                    x, y = pygame.mouse.get_pos()
                    pygame.draw.circle(tela, cor_pincel,(x, y), 22)
                    for event2 in pygame.event.get():
                        if event2.type == pygame.MOUSEBUTTONUP:
                            pintar = False
                            break
                break
            if event.type == pygame.QUIT:
                sair = True
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    pygame.image.save(tela, 'tela.jpeg')
                    img = cv2.imread('tela.jpeg')
                    res = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
                    x = []
                    for c in res:
                        for j in c:
                            x.append(j[0]*0.003)
                    x = torch.Tensor(x).to(args['device'])
                    x = x.reshape(1, 32, 32)
                    x = x.unsqueeze(1)
                    x = x.to(args['device'])
                    pred = net(x)
                    pred = pred.data.long().numpy()
                    resultado = np.where(pred == pred.max())[1][0]
                    img = font.render(f'{resultado} ', True, 'white')
                    tela.blit(img, (10, 10))
                if event.key == pygame.K_UP:
                    tela = pygame.display.set_mode([largura, altura])
                    tela.fill(cor_branca)
        relogio.tick(frames)
        pygame.display.update()
    pygame.quit()



if __name__ == '__main__':
    with open('redeneural.pkl','rb') as f:
        net = pickle.load(f)
    main(frames)
