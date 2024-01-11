import torch
from braid.models import get_the_resnet_model


def load_trained_model(model_name, mlp_hidden_layer_sizes, feature_vector_length, path_pth, device='cuda'):
    # model architecture
    model = get_the_resnet_model(
        model_name = model_name,
        feature_vector_length = feature_vector_length,
        MLP_hidden_layer_sizes = mlp_hidden_layer_sizes,
    )
    
    # load model weights
    checkpoint = torch.load(path_pth)
    model.load_state_dict(checkpoint)
    
    if device == 'cuda':
        model = model.to(torch.device('cuda'))
    
    print(f"Trained model loaded in {device}")
    return model
