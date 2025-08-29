"""
Script simple de inferencia para DenseNet MNIST
Solo cambia la ruta de la imagen y el modelo en las variables al inicio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ========================= CONFIGURACIÓN =========================
IMAGE_PATH = "image.png"  # 
MODEL_PATH = "densenet_mnist_fixed_final.pth" 
# ================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Arquitectura del modelo (debe coincidir exactamente con el entrenamiento)
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 
                              kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate, 
                             bn_size, drop_rate)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels,
                             kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        out = self.pool(out)
        return out

class SimpleDenseNet(nn.Module):
    def __init__(self, growth_rate=8, block_config=(4, 6, 4),
                 num_init_features=16, bn_size=2, drop_rate=0.2, num_classes=10):
        super(SimpleDenseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, in_channels=num_features,
                             bn_size=bn_size, growth_rate=growth_rate,
                             drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = Transition(in_channels=num_features,
                                 out_channels=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def load_model():
    """Cargar el modelo entrenado"""
    model = SimpleDenseNet()
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"✅ Modelo cargado desde: {MODEL_PATH}")
        return model
    
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return None

def preprocess_image():
    """Cargar y preprocesar la imagen"""
    try:
        # Cargar imagen
        image = Image.open(IMAGE_PATH)
        print(f"📷 Imagen cargada: {image.size}, modo: {image.mode}")
        
        # Convertir a escala de grises
        if image.mode != 'L':
            image = image.convert('L')
        
        # Redimensionar a 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(image)
        
        # Invertir colores si es necesario (MNIST tiene fondo negro)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
            print("🔄 Colores invertidos")
        
        # Normalizar
        img_array = img_array.astype(np.float32) / 255.0
        
        # Convertir a tensor y normalizar con estadísticas MNIST
        transform = transforms.Normalize((0.1307,), (0.3081,))
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        img_tensor = transform(img_tensor)
        
        return img_tensor, img_array
    
    except Exception as e:
        print(f"❌ Error al procesar imagen: {e}")
        return None, None

def predict(model, image_tensor):
    """Hacer predicción"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]

def show_results(image_array, prediction, confidence, probabilities):
    """Mostrar resultados de forma simple"""
    # Crear visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mostrar imagen
    ax1.imshow(image_array, cmap='gray')
    ax1.set_title(f'Imagen de Entrada', fontsize=14)
    ax1.axis('off')
    
    # Mostrar probabilidades
    digits = range(10)
    colors = ['red' if i == prediction else 'lightblue' for i in digits]
    bars = ax2.bar(digits, probabilities, color=colors, alpha=0.8)
    
    # Destacar la predicción
    bars[prediction].set_color('red')
    
    ax2.set_xlabel('Dígito')
    ax2.set_ylabel('Probabilidad')
    ax2.set_title(f'Predicción: {prediction} (Confianza: {confidence:.3f})')
    ax2.set_xticks(digits)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for i, prob in enumerate(probabilities):
        ax2.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold' if i == prediction else 'normal')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar resultados en texto
    print(f"\n{'='*50}")
    print(f"🎯 PREDICCIÓN: {prediction}")
    print(f"📊 CONFIANZA: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"{'='*50}")
    
    # Top 3 predicciones
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    print(f"\n🏆 TOP 3 PREDICCIONES:")
    for i, idx in enumerate(top3_indices):
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"{emoji} Dígito {idx}: {probabilities[idx]:.4f} ({probabilities[idx]*100:.2f}%)")

def main():
    """Función principal"""
    print("🚀 INICIANDO PREDICCIÓN DE DÍGITO")
    print("=" * 50)
    
    # Cargar modelo
    model = load_model()
    if model is None:
        return
    
    # Procesar imagen
    image_tensor, image_array = preprocess_image()
    if image_tensor is None:
        return
    
    # Hacer predicción
    print("🧠 Realizando predicción...")
    prediction, confidence, probabilities = predict(model, image_tensor)
    
    # Mostrar resultados
    show_results(image_array, prediction, confidence, probabilities)
    
    print(f"\n✅ Predicción completada!")

if __name__ == "__main__":
    main()