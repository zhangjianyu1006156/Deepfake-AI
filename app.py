import multiprocessing
import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
from AI_proj import RESNET_LSTM_for_DeepFake, augmented_transform

# Function to load the model, using Streamlit's caching to avoid reloading the model on each interaction
@st.cache(allow_output_mutation=True)
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RESNET_LSTM_for_DeepFake(num_classes=2).to(device)
    model.load_state_dict(torch.load('model_state.pth', map_location=device))
    model.eval()  # Set the model to evaluation mode.
    return model

# Function to plot accuracies and losses
def plot_metrics(train_metric, val_metric, test_metric, title, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(train_metric, label='Train')
    plt.plot(val_metric, label='Validation')
    plt.plot(test_metric, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def main():
    st.title('RESNET LSTM for Lung Cancer Detection')

    model = load_model()

    st.header('Image Transformations')
    # Assuming the images are preloaded or available in a directory
    # Displaying original and augmented images as example
    raw_dataset = datasets.ImageFolder(root='processed_dataset_frame/processed_dataset_frame', transform=None)
    raw_img, label = raw_dataset[0]

    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_img, caption='Original Image')

    with col2:
        # Apply augmented transformation and convert tensor to image
        augmented_img = transforms.ToPILImage()(augmented_transform(raw_img)).convert("RGB")
        st.image(augmented_img, caption='Augmented Image')

    # Training Metrics Section
    st.header('Training Metrics')
    # Load the metrics
    train_accuracies = np.load('train_accuracies.npy')
    validation_accuracies = np.load('validation_accuracies.npy')
    test_accuracies = np.load('test_accuracies.npy')

    st.subheader('Accuracies')
    plot_metrics(train_accuracies, validation_accuracies, test_accuracies, 'Accuracies', 'Accuracy (%)')

    # Load the losses
    train_losses = np.load('train_losses.npy')
    validation_losses = np.load('validation_losses.npy')
    test_losses = np.load('test_losses.npy')

    st.subheader('Losses')
    plot_metrics(train_losses, validation_losses, test_losses, 'Losses', 'Loss')

    st.header('Make Your Own Prediction')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Transform the image to tensor
        transformed_image = augmented_transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            model.eval()  # Ensure model is in evaluation mode
            logits = model(transformed_image)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = probabilities.argmax(1).item()

        # Assuming you have a list or dict of class names
        class_names = ['Benign Case', 'Malignant Case', 'Normal Case']  # Update with your actual class names
        predicted_class_name = class_names[predicted_class]

        # Show the prediction
        st.write(f"Predicted Class: {predicted_class_name} with a probability of {probabilities[0][predicted_class].item()*100:.2f}%")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Good for Windows executable
    main()  # Your main function which might set up DataLoaders, etc.

