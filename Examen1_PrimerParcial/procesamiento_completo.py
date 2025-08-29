import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
import glob
from collections import OrderedDict
import matplotlib.pyplot as plt

# ==================== CONFIGURACIÓN ====================
RUTA_RECORTES = "Recortes_votos"
MODELO_CODIGOS = "lightweight_ocr_model.pth"
MODELO_VOTOS = "densenet_mnist_fixed_final.pth"
CSV_RESULTADOS = "resultados_finales.csv"
# ======================================================

# Importar las arquitecturas de tus modelos (copiar desde tus archivos)
class LightweightOCR(nn.Module):
    def __init__(self, num_classes=11, max_length=8):
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

    pass


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
    pass

# ==================== FUNCIONES DE INFERENCIA ====================

def cargar_modelo_codigos():
    """Cargar modelo para códigos de mesa"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(MODELO_CODIGOS, map_location=device)
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    num_classes = checkpoint['num_classes']
    max_length = checkpoint['max_length']
    
    model = LightweightOCR(num_classes=num_classes, max_length=max_length)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, idx_to_char, char_to_idx, device

def cargar_modelo_votos():
    """Cargar modelo para dígitos manuscritos"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleDenseNet()
    checkpoint = torch.load(MODELO_VOTOS, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, device

def predecir_codigo_mesa(model, idx_to_char, device, imagen_path):
    """Predecir código de mesa usando tu código de inferencia_codigos.py"""
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    original_image = Image.open(imagen_path)
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        predictions = torch.argmax(outputs, dim=2)
    
    pred_chars = []
    for pred_idx in predictions[0]:
        if pred_idx.item() < len(idx_to_char):
            char = idx_to_char[pred_idx.item()]
            pred_chars.append(char)
    
    return ''.join(pred_chars)

def predecir_digito_manuscrito(model, device, imagen_path):
    """Predecir dígito manuscrito usando tu código de inferencia_votos.py"""
    # Cargar y preprocesar imagen
    image = Image.open(imagen_path)
    if image.mode != 'L':
        image = image.convert('L')
    
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    # Invertir colores si es necesario
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Normalizar
    img_array = img_array.astype(np.float32) / 255.0
    transform = transforms.Normalize((0.1307,), (0.3081,))
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    img_tensor = transform(img_tensor)
    
    # Predecir
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

# ==================== PROCESAMIENTO COMPLETO ====================

def procesar_acta_completa(ruta_acta, modelo_codigos, idx_to_char, device_codigos, modelo_votos, device_votos):
    """Procesar una acta completa y devolver todos los resultados"""
    resultados = {}
    
    # 1. Procesar código de mesa
    codigo_path = os.path.join(ruta_acta, 'codigo_mesa.jpg')
    if os.path.exists(codigo_path):
        try:
            codigo = predecir_codigo_mesa(modelo_codigos, idx_to_char, device_codigos, codigo_path)
            resultados['codigo_mesa'] = codigo
        except Exception as e:
            print(f"❌ Error procesando código de mesa: {e}")
            resultados['codigo_mesa'] = 'ERROR'
    else:
        resultados['codigo_mesa'] = 'NO_ENCONTRADO'
    
    # 2. Procesar todos los tipos de votos
    categorias_votos = [
        'votos_AP', 'votos_LYP_ADN', 'votos_APB_SUMATE',
        'votos_LIBRE', 'votos_FP', 'votos_MAS_IPSP',
        'votos_UNIDAD', 'votos_PDC', 'votos_validos',
        'votos_blancos', 'votos_nulos'
    ]
    
    for categoria in categorias_votos:
        carpeta_categoria = os.path.join(ruta_acta, categoria)
        
        if os.path.exists(carpeta_categoria):
            try:
                # Predecir cada dígito (centenas, decenas, unidades)
                centenas_path = os.path.join(carpeta_categoria, f'{categoria}_centenas.jpg')
                decenas_path = os.path.join(carpeta_categoria, f'{categoria}_decenas.jpg')
                unidades_path = os.path.join(carpeta_categoria, f'{categoria}_unidades.jpg')
                
                centenas, _ = predecir_digito_manuscrito(modelo_votos, device_votos, centenas_path)
                decenas, _ = predecir_digito_manuscrito(modelo_votos, device_votos, decenas_path)
                unidades, _ = predecir_digito_manuscrito(modelo_votos, device_votos, unidades_path)
                
                # Formar número completo
                numero_completo = f"{centenas}{decenas}{unidades}"
                resultados[categoria] = int(numero_completo)
                
            except Exception as e:
                print(f"❌ Error procesando {categoria}: {e}")
                resultados[categoria] = -1  # Valor de error
        else:
            resultados[categoria] = -2  # No encontrado
    
    return resultados

def main():
    """Función principal"""
    print("🚀 INICIANDO PROCESAMIENTO COMPLETO DE ACTAS")
    print("=" * 60)
    
    # Cargar modelos
    print("📦 Cargando modelos...")
    modelo_codigos, idx_to_char, char_to_idx, device_codigos = cargar_modelo_codigos()
    modelo_votos, device_votos = cargar_modelo_votos()
    print("✅ Modelos cargados correctamente")
    
    # Encontrar todas las actas
    actas_paths = [f for f in os.listdir(RUTA_RECORTES) 
                  if os.path.isdir(os.path.join(RUTA_RECORTES, f))]
    
    print(f"📁 Encontradas {len(actas_paths)} actas para procesar")
    
    # Procesar cada acta
    todos_resultados = []
    
    for i, acta_folder in enumerate(actas_paths, 1):
        print(f"\n🔨 Procesando acta {i}/{len(actas_paths)}: {acta_folder}")
        
        ruta_acta_completa = os.path.join(RUTA_RECORTES, acta_folder)
        resultados = procesar_acta_completa(
            ruta_acta_completa, modelo_codigos, idx_to_char, device_codigos, 
            modelo_votos, device_votos
        )
        
        resultados['acta_folder'] = acta_folder
        todos_resultados.append(resultados)
        
        if i % 10 == 0:
            print(f"   ✅ {i} actas procesadas...")
    
    # Crear DataFrame y guardar CSV
    df = pd.DataFrame(todos_resultados)
    
    # Reordenar columnas
    column_order = ['acta_folder', 'codigo_mesa'] + [
        'votos_AP', 'votos_LYP_ADN', 'votos_APB_SUMATE',
        'votos_LIBRE', 'votos_FP', 'votos_MAS_IPSP',
        'votos_UNIDAD', 'votos_PDC', 'votos_validos',
        'votos_blancos', 'votos_nulos'
    ]
    
    df = df[column_order]
    df.to_csv(CSV_RESULTADOS, index=False)
    
    print(f"\n🎉 PROCESAMIENTO COMPLETADO!")
    print(f"📊 Resultados guardados en: {CSV_RESULTADOS}")
    print(f"📋 Total de actas procesadas: {len(todos_resultados)}")
    
    # Mostrar resumen
    print("\n📈 RESUMEN FINAL:")
    print(df.head())

if __name__ == "__main__":
    main()