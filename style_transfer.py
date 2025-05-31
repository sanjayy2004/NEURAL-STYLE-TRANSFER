import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess image
def load_image(path, max_size=400):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Show tensor as image
def imshow(tensor, title=None):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title: 
        plt.title(title)
        plt.axis('off')
        plt.show()

# Gram matrix
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)

# Get features from VGG
def get_features(image, model, layers):
    features = {}
    x = image
    i = 0
    for layer in model.children():
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
            if name in layers:
                features[name] = x
    return features

# Load content and style images
content = load_image("content.jpg")
style = load_image("style.jpg")
generated = content.clone().requires_grad_(True)

# Load VGG19
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Define layers to use
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Get content and style features
content_features = {layer: tensor.detach() for layer, tensor in get_features(content, vgg, content_layers).items()}

style_features = get_features(style, vgg, style_layers)

# Calculate style Gram matrices (detached)
style_grams = {layer: gram_matrix(style_features[layer]).detach() for layer in style_layers}

# Hyperparameters
style_weight = 1e6
content_weight = 1e0
optimizer = optim.Adam([generated], lr=0.003)

# Training loop
for step in range(1, 301):
    optimizer.zero_grad()
    
    gen_features = get_features(generated, vgg, style_layers + content_layers)
    
    # Content loss
    content_loss = torch.mean((gen_features['conv_4'] - content_features['conv_4']) ** 2)

    
    # Style loss
    style_loss = 0
    for layer in style_layers:
        gen_feature = gen_features[layer]
        gen_gram = gram_matrix(gen_feature)
        style_gram = style_grams[layer]
        layer_loss = torch.mean((gen_gram - style_gram) ** 2)
        style_loss += layer_loss

    total_loss = content_weight * content_loss + style_weight * style_loss
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Total Loss: {total_loss.item():.4f}")
        imshow(generated, title=f"Step {step}")

# Save final result
final = generated.clone().detach().cpu().squeeze(0)
final_image = transforms.ToPILImage()(final)
final_image.save("output.jpg")
print("✅ Style transfer complete. Saved as output.jpg")
