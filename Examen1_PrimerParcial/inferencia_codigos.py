import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

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

def load_model(model_path):
    """Cargar modelo entrenado"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extraer información
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    num_classes = checkpoint['num_classes']
    max_length = checkpoint['max_length']
    
    # Crear modelo
    model = LightweightOCR(num_classes=num_classes, max_length=max_length)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Modelo cargado en {device}")
    if 'test_accuracy' in checkpoint:
        print(f"✓ Precisión del modelo: {checkpoint['test_accuracy']:.4f}")
    
    return model, idx_to_char, char_to_idx, device

def predict_image(model, image_path, idx_to_char, device):
    """Predecir código de una imagen"""
    
    # Transformaciones (mismas que en entrenamiento)
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Cargar imagen original
    original_image = Image.open(image_path)
    print(f"✓ Imagen cargada: {original_image.size}")
    
    # Convertir a RGB si es necesario
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    
    # Preprocesar para el modelo
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Hacer predicción
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=2)
        predictions = torch.argmax(outputs, dim=2)
    
    # Decodificar predicción
    pred_chars = []
    confidences = []
    
    for i, pred_idx in enumerate(predictions[0]):
        if pred_idx.item() < len(idx_to_char):
            char = idx_to_char[pred_idx.item()]
            conf = probabilities[0, i, pred_idx].item()
            pred_chars.append(char)
            confidences.append(conf)
    
    predicted_text = ''.join(pred_chars)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return original_image, predicted_text, avg_confidence, confidences

def test_image(image_path, model_path="lightweight_ocr_model.pth", show_image=True):
    """Función principal para probar una imagen"""
    
    # Verificar que la imagen existe
    if not os.path.exists(image_path):
        print(f"❌ Error: La imagen no existe en la ruta: {image_path}")
        return
    
    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        print(f"❌ Error: El modelo no existe en la ruta: {model_path}")
        print("Asegúrate de haber entrenado el modelo primero.")
        return
    
    try:
        print(f"🔍 Probando imagen: {image_path}")
        print(f"🤖 Usando modelo: {model_path}")
        print("-" * 50)
        
        # Cargar modelo
        model, idx_to_char, char_to_idx, device = load_model(model_path)
        
        # Hacer predicción
        original_image, prediction, confidence, char_confidences = predict_image(
            model, image_path, idx_to_char, device
        )
        
        # Mostrar resultados
        print(f"\n📋 RESULTADOS:")
        print(f"🎯 Predicción: '{prediction}'")
        print(f"📊 Confianza promedio: {confidence:.4f} ({confidence*100:.1f}%)")
        
        print(f"\n📈 Confianza por carácter:")
        for i, (char, conf) in enumerate(zip(prediction, char_confidences)):
            print(f"   Posición {i+1}: '{char}' -> {conf:.4f} ({conf*100:.1f}%)")
        
        # Mostrar imagen si se solicita
        if show_image:
            plt.figure(figsize=(12, 4))
            
            # Imagen original
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title('Imagen Original')
            plt.axis('off')
            
            # Imagen redimensionada (como la ve el modelo)
            resized = original_image.resize((128, 32))
            plt.subplot(1, 2, 2)
            plt.imshow(resized)
            plt.title(f'Imagen Procesada\nPredicción: "{prediction}"')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return prediction, confidence
        
    except Exception as e:
        print(f"❌ Error durante la predicción: {str(e)}")
        return None, None

# Función para probar múltiples imágenes
def test_multiple_images(image_paths, model_path="lightweight_ocr_model.pth"):
    """Probar múltiples imágenes"""
    
    if not os.path.exists(model_path):
        print(f"❌ Error: El modelo no existe en la ruta: {model_path}")
        return
    
    # Cargar modelo una sola vez
    print("🤖 Cargando modelo...")
    model, idx_to_char, char_to_idx, device = load_model(model_path)
    
    results = []
    
    print(f"\n🔍 Probando {len(image_paths)} imágenes...")
    print("-" * 50)
    
    for i, image_path in enumerate(image_paths, 1):
        if not os.path.exists(image_path):
            print(f"❌ Imagen {i}: No existe - {image_path}")
            continue
            
        try:
            _, prediction, confidence, _ = predict_image(model, image_path, idx_to_char, device)
            results.append({
                'path': image_path,
                'prediction': prediction,
                'confidence': confidence
            })
            print(f"✓ Imagen {i}: '{prediction}' (conf: {confidence:.3f}) - {os.path.basename(image_path)}")
            
        except Exception as e:
            print(f"❌ Imagen {i}: Error - {str(e)} - {os.path.basename(image_path)}")
    
    return results

if __name__ == "__main__":
    # ==========================================
    # CONFIGURACIÓN - CAMBIA ESTAS RUTAS
    # ==========================================
    
    # Ruta de tu imagen (CAMBIA ESTA RUTA)
    IMAGE_PATH = "imagen.jpg"  # Cambia por la ruta de tu imagen
    
    # Ruta del modelo entrenado (opcional, usa el nombre por defecto si no cambias)
    MODEL_PATH = "lightweight_ocr_model.pth"
    
    # ==========================================
    # EJECUCIÓN
    # ==========================================
    
    print("🚀 PROBADOR DE MODELO OCR")
    print("=" * 50)
    
    # Probar una sola imagen
    prediction, confidence = test_image(IMAGE_PATH, MODEL_PATH, show_image=True)
    
    # Ejemplo de cómo probar múltiples imágenes (descomenta si necesitas)
    """
    image_list = [
        "imagen1.png",
        "imagen2.png", 
        "imagen3.png"
    ]
    results = test_multiple_images(image_list, MODEL_PATH)
    """
    
    print("\n✅ ¡Prueba completada!")