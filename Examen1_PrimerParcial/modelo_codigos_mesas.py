import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing

# Configurar multiprocessing para Windows
if __name__ == '__main__':
    multiprocessing.freeze_support()

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')

# Configuración de parámetros
BATCH_SIZE = 64  # Incrementamos batch size
LEARNING_RATE = 0.001
EPOCHS = 1  # Reducimos epochs
DATASET_PATH = "synthetic_dataset"  # Cambia por la ruta a tu dataset
MODEL_SAVE_PATH = "lightweight_ocr_model.pth"

# Caracteres posibles en nuestro dataset
CHARS = '0123456789-'
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)
MAX_LENGTH = 8  # Longitud máxima del código (NNNNNN-N)

class CodeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.labels_dir = os.path.join(root_dir, split, 'labels')
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Cargar etiqueta
        label_name = img_name.replace('.png', '.txt')
        label_path = os.path.join(self.labels_dir, label_name)
        with open(label_path, 'r') as f:
            label = f.read().strip()
        
        # Convertir etiqueta a secuencia de índices
        label_indices = [CHAR_TO_IDX[char] for char in label]
        
        # Padding si es necesario
        while len(label_indices) < MAX_LENGTH:
            label_indices.append(NUM_CLASSES)  # Usar NUM_CLASSES como padding
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label_indices, dtype=torch.long), label

# Transformaciones más simples
transform_train = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalización más simple
])

transform_val = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Cargar datasets
train_dataset = CodeDataset(DATASET_PATH, 'train', transform_train)
val_dataset = CodeDataset(DATASET_PATH, 'val', transform_val)
test_dataset = CodeDataset(DATASET_PATH, 'test', transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'Dataset cargado:')
print(f'Train: {len(train_dataset)} imágenes')
print(f'Val: {len(val_dataset)} imágenes')
print(f'Test: {len(test_dataset)} imágenes')

class LightweightOCR(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, max_length=MAX_LENGTH):
        super(LightweightOCR, self).__init__()
        
        # Backbone CNN muy liviano y eficiente
        self.feature_extractor = nn.Sequential(
            # Primer bloque
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x64
            
            # Segundo bloque
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x32
            
            # Tercer bloque
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 4x32 - mantener más ancho
            
            # Cuarto bloque
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 2x32
            
            # Quinto bloque
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 1x32
        )
        
        # LSTM más pequeño
        self.lstm_input_size = 512
        self.lstm_hidden_size = 128
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,  # Solo 1 capa
            batch_first=True,
            bidirectional=True  # Removido dropout para 1 capa
        )
        
        # Clasificador más simple
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2, 256),  # *2 por bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes + 1)  # +1 for padding
        )
        
        self.max_length = max_length
        self.num_classes = num_classes + 1
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extraer características CNN
        features = self.feature_extractor(x)  # [batch, 512, 1, 32]
        
        # Reformatear para LSTM: (batch, seq_len, features)
        features = features.squeeze(2)  # [batch, 512, 32]
        features = features.permute(0, 2, 1)  # [batch, 32, 512]
        
        # LSTM
        lstm_out, _ = self.lstm(features)  # [batch, 32, 256]
        
        # Tomar solo las posiciones relevantes (8 caracteres)
        # Interpolar para obtener exactamente MAX_LENGTH posiciones
        if lstm_out.size(1) != self.max_length:
            lstm_out = nn.functional.interpolate(
                lstm_out.permute(0, 2, 1), 
                size=self.max_length, 
                mode='linear', 
                align_corners=False
            ).permute(0, 2, 1)
        
        # Clasificación para cada posición
        output = self.classifier(lstm_out)  # [batch, max_length, num_classes]
        
        return output

# Crear modelo
model = LightweightOCR().to(device)

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Parámetros totales: {total_params:,}')
print(f'Parámetros entrenables: {trainable_params:,}')

# Criterio de pérdida y optimizador
criterion = nn.CrossEntropyLoss(ignore_index=NUM_CLASSES)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=LEARNING_RATE * 5,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader)
)

def calculate_accuracy(outputs, targets):
    """Calcular precisión a nivel de secuencia completa"""
    batch_size = outputs.size(0)
    predicted = torch.argmax(outputs, dim=2)
    
    # Comparar secuencias completas
    correct = 0
    for i in range(batch_size):
        pred_seq = predicted[i]
        target_seq = targets[i]
        
        # Remover padding
        pred_chars = [p.item() for p in pred_seq if p.item() < NUM_CLASSES]
        target_chars = [t.item() for t in target_seq if t.item() < NUM_CLASSES]
        
        if pred_chars == target_chars:
            correct += 1
    
    return correct / batch_size

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, targets, _) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Reshape para pérdida
        outputs_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = targets.view(-1)
        
        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Calcular métricas
        total_loss += loss.item()
        accuracy = calculate_accuracy(outputs, targets)
        total_accuracy += accuracy
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.4f}',
            'LR': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    return total_loss / len(loader), total_accuracy / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, targets, _ in pbar:
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            
            # Reshape para pérdida
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            
            accuracy = calculate_accuracy(outputs, targets)
            total_accuracy += accuracy
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}'
            })
    
    return total_loss / len(loader), total_accuracy / len(loader)

def decode_prediction(prediction):
    """Decodificar predicción a texto"""
    chars = []
    for idx in prediction:
        idx_int = idx.item()  # Convertir tensor a int
        if idx_int < NUM_CLASSES:
            chars.append(IDX_TO_CHAR[idx_int])
    return ''.join(chars)

if __name__ == '__main__':
    print("Iniciando entrenamiento...")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 30)
        
        # Entrenamiento
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        
        # Validación
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Guardar métricas
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Epoch Time: {epoch_time:.1f}s')
        
        # Guardar mejor modelo
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict().copy()
            print(f'Nuevo mejor modelo guardado! Precisión: {val_acc:.4f}')

    total_time = time.time() - start_time
    print(f'\nTiempo total de entrenamiento: {total_time/60:.1f} minutos')

    # Cargar mejor modelo para evaluación final
    model.load_state_dict(best_model_state)
    print(f'\nMejor precisión de validación: {best_val_accuracy:.4f}')

    # Evaluación en test set
    print('\nEvaluando en conjunto de test...')
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    # Mostrar algunas predicciones de ejemplo
    model.eval()
    with torch.no_grad():
        images, targets, labels = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=2)
        
        print('\nEjemplos de predicciones:')
        for i in range(min(10, len(images))):
            pred_text = decode_prediction(predictions[i])
            true_text = labels[i]
            status = '✓' if pred_text == true_text else '✗'
            print(f'Real: {true_text:10} | Predicción: {pred_text:10} | {status}')

    # Guardar modelo final
    torch.save({
        'model_state_dict': best_model_state,
        'char_to_idx': CHAR_TO_IDX,
        'idx_to_char': IDX_TO_CHAR,
        'num_classes': NUM_CLASSES,
        'max_length': MAX_LENGTH,
        'test_accuracy': test_acc,
        'training_time_minutes': total_time/60,
        'model_config': {
            'input_size': (32, 128),
            'num_classes': NUM_CLASSES,
            'max_length': MAX_LENGTH,
            'total_params': total_params
        }
    }, MODEL_SAVE_PATH)

    print(f'\nModelo guardado como: {MODEL_SAVE_PATH}')
    print(f'Precisión final en test: {test_acc:.4f}')
    print(f'Parámetros del modelo: {total_params:,}')

    # Graficar curvas de entrenamiento
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss durante entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Accuracy durante entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves_lightweight.png')
    plt.show()

    print('\n¡Entrenamiento completado!')
    print(f'Tiempo promedio por época: {(total_time/EPOCHS)/60:.1f} minutos')
    
