import torch
checkpoint = torch.load('models/best_model_fold5.pt', map_location='cpu')
for key in list(checkpoint.keys())[:20]:
    print(f"{key}: {checkpoint[key].shape}")