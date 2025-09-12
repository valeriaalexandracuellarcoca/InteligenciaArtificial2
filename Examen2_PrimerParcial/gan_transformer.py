#importamos las librerias necesarias
import torch
import torchvision
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import torch.nn as nn
from fastprogress import master_bar, progress_bar
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

#cargar el dataset FashionMNIST que ya estaba descargado
trainset = torchvision.datasets.FashionMNIST(root='MNIST/data', train=True, download=False)
#clases del dataset

classes = ("t-shirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot")

#
class Dataset(torch.utils.data.Dataset):
  def __init__(self, trainset):
    imgs_list = [np.array(i[0]) for i in trainset]  # Lista de (1,28,28) ya normalizados
    self.imgs = torch.tensor(np.stack(imgs_list), dtype=torch.float, device=device)  # (N,1,28,28)
    self.labels = torch.tensor([i[1] for i in trainset], dtype=torch.long, device=device) #etiquetas
  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, ix):
    return self.imgs[ix], self.labels[ix]

train = Dataset(trainset) #instancia del dataset
len(train) #tamaño del dataset  

dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True) #dataloader

#clase para MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.n_heads = n_heads

        # Proyecciones para key, query, value
        self.key = nn.Linear(n_embd, n_embd * n_heads) 
        self.query = nn.Linear(n_embd, n_embd * n_heads)
        self.value = nn.Linear(n_embd, n_embd * n_heads)

        #proyección de salida
        self.proj = nn.Linear(n_embd * n_heads, n_embd)

    def forward(self, x):
        B, L, F = x.size()

        #calculo de query, key, values para todas las cabezas
        k = self.key(x).view(B, L, F, self.n_heads).transpose(1, 3)  # (B, nh, L, F)
        q = self.query(x).view(B, L, F, self.n_heads).transpose(1, 3)  # (B, nh, L, F)
        v = self.value(x).view(B, L, F, self.n_heads).transpose(1, 3)  # (B, nh, L, F)

        #atención: (B, nh, L, F) x (B, nh, F, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #aplica la fórmula de atención 
        att = nn.functional.softmax(att, dim=-1)
        #aplicar atención a los valores
        y = att @ v  # (B, nh, L, L) x (B, nh, L, F) -> (B, nh, L, F)
        y = y.transpose(1, 2).contiguous().view(B, L, F * self.n_heads)  #reensamblar cabezas

        return self.proj(y)

#bloque Transformer
class BloqueTransformer(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.mlp(x))
        return x

#Generador basado en un Transformer
class GeneradorTransfomer(nn.Module):
    def __init__(self, latent_dim=100, patch_dim=49, seq_len=16, n_embd=128, n_heads=8, n_layers=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.patch_size = 7  #para parches de 7x7
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, n_embd))
        self.proj_in = nn.Linear(latent_dim, seq_len * n_embd)
        self.transformer = nn.Sequential(*[BloqueTransformer(n_embd, n_heads) for _ in range(n_layers)])
        self.proj_out = nn.Linear(n_embd, patch_dim)

    def forward(self, z):
        B = z.size(0)
        #proyectar ruido a secuencia de embeddings
        e = self.proj_in(z).view(B, self.seq_len, -1) + self.pos_emb
        #aplicar bloques Transformer
        x = self.transformer(e)
        #proyectar a parches
        patches = self.proj_out(x)  # (B, seq_len, patch_dim)
        #reconstruir la imagen de 28x28
        patches = patches.view(B, 4, 4, self.patch_size, self.patch_size)  # 4x4 grid de parches
        img = torch.zeros(B, 28, 28, device=z.device)
        for i in range(4):
            for j in range(4):
                img[:, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size] = patches[:, i, j]
        return img.view(B, -1)  # Aplanar a (B, 784)

#bloques MLP(para usar en el discriminador)
def bloque(n_in, n_out):
  return nn.Sequential(
      nn.Linear(n_in, n_out),
      nn.ReLU(inplace=True)
  )

#clase MLP para el Discriminador
class MLP(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.fc1 = bloque(input_size, 150)
    self.fc2 = bloque(150, 100)
    self.fc3 = nn.Linear(100, output_size)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x

#instanciar generador y discriminador
latent_dim = 100 #define la dimensionalidad del vector de ruido aleatorio que sirve como entrada al generador
generator = GeneradorTransfomer(latent_dim)
discriminator = MLP(28*28, 1)

#fuunción de entrenamiento
def fit(g, d, dataloader, epochs=5, crit=None):   
    g.to(device)   
    d.to(device)   
    
    #optimizador Adam para el generador con learning rate de 3e-4
    g_optimizer = torch.optim.Adam(g.parameters(), lr=3e-4)   
    #optimizador Adam para el discriminador con learning rate de 3e-4
    d_optimizer = torch.optim.Adam(d.parameters(), lr=3e-4)   
    
    #usar BCEWithLogitsLoss por defecto
    crit = nn.BCEWithLogitsLoss() if crit is None else crit   
    
    #arrays para almacenar las perdidas
    g_loss, d_loss = [], []   
    
    #barra de progreso
    mb = master_bar(range(1, epochs+1))   
    
    #diccionario para almacenar el historial de pérdidas
    hist = {'g_loss': [], 'd_loss': []}   
    #bucle principal
    for epoch in mb:     
        #bucle sobre todos los batches del dataloader
        for X, y in progress_bar(dataloader, parent=mb):       
            #ENTRENAMIENTO DEL DISCRIMINADOR
            #poner el generador en modo evaluación(sin gradientes)
            g.eval()       
            #poner el discriminador en modo entrenamiento
            d.train()       
            
            #generar ruido aleatorio con la misma cantidad de muestras que el batch
            noise = torch.randn((X.size(0), g.latent_dim)).to(device)       
            #generar imágenes falsas usando el generador
            generated_images = g(noise)       
            
            #concatenar imgenes generadas con imágenes reales aplanadas
            d_input = torch.cat([generated_images, X.view(X.size(0), -1)])       
            d_gt = torch.cat([torch.zeros(X.size(0)), torch.ones(X.size(0))]).view(-1, 1).to(device)       
            

            d_optimizer.zero_grad()       
            #predicciones del discriminador
            d_output = d(d_input)       
            #pérdida del discriminador
            d_l = crit(d_output, d_gt)       
            d_l.backward()       
            d_optimizer.step()       
            d_loss.append(d_l.item())       
            
            #ENTRENAMIENTO DEL GENERADOR
            # Poner el generador en modo entrenamiento
            g.train()       
            # Poner el discriminador en modo evaluación(sin actualizar sus pesos)
            d.eval()       
            
            #generar nuevo ruido aleatorio
            noise = torch.randn((X.size(0), g.latent_dim)).to(device)       
            #generar nuevas imágenes falsas
            generated_images = g(noise)       
            #pasar las imágenes generadas por el discriminador
            d_output = d(generated_images)       
            #crear etiquetas de reales para engañar al discriminador
            g_gt = torch.ones(X.size(0)).view(-1, 1).to(device)       
            
            #limpiar gradientes del generador
            g_optimizer.zero_grad()       
            #pérdida del generador
            g_l = crit(d_output, g_gt)       
            #Retropropagación de la pérdida
            g_l.backward()       
            #Actualizar parámetros del generador
            g_optimizer.step()       
            #Guardar la pérdida del generador
            g_loss.append(g_l.item())       
            
            # Actualizar comentario de la barra de progreso con pérdidas promedio
            mb.child.comment = f'g_loss {np.mean(g_loss):.5f} d_loss {np.mean(d_loss):.5f}'     
        
        mb.write(f'Epoch {epoch}/{epochs} g_loss {np.mean(g_loss):.5f} d_loss {np.mean(d_loss):.5f}')     
        #guardar pérdidas promedio de la época en el historial
        hist['g_loss'].append(np.mean(g_loss))     
        hist['d_loss'].append(np.mean(d_loss))   
    
    return hist

#llamar a la funcion que entrena la GAN
hist = fit(generator, discriminator, dataloader)

#perdidas
df = pd.DataFrame(hist)
df.plot(grid=True)
plt.show()

#generar imgenes
generator.eval()
with torch.no_grad():
  noise = torch.randn((10, latent_dim)).to(device)
  generated_images = generator(noise)
  fig, axs = plt.subplots(2, 5, figsize=(15, 5))
  i = 0
  for ax in axs:
    for _ax in ax:
      img = generated_images[i].view(28, 28).cpu()
      _ax.imshow(img)
      i += 1
  plt.show()