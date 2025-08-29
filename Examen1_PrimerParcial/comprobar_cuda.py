import torch

# Verifica si CUDA está disponible
cuda_available = torch.cuda.is_available()
print(f"¿CUDA está disponible?: {cuda_available}")

if cuda_available:
    # Obtiene el número de GPUs disponibles
    gpu_count = torch.cuda.device_count()
    print(f"Número de GPUs disponibles: {gpu_count}")
    
    # Muestra información de cada GPU
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memoria total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("No se detectó una GPU compatible o CUDA no está instalado correctamente.")