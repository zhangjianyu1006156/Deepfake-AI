import streamlit as st
import cv2
import numpy as np
import glob
import os
import torch
from torchvision import transforms
from torchvision.models import resnext50_32x4d  # Assuming you're using this architecture
import gc

# Function to save frames from video
def save_frames_from_video(video_path, frames_dir, max_frames=50):
    vidObj = cv2.VideoCapture(video_path)
    total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // max_frames)
    frames_saved = 0

    frame_index = 0
    success, frame = vidObj.read()
    while success and frames_saved < max_frames:
        if frame_index % frame_step == 0:
            frame_file_path = os.path.join(frames_dir, f"frame_{frame_index}.jpg")
            cv2.imwrite(frame_file_path, frame)
            frames_saved += 1
        success, frame = vidObj.read()
        frame_index += 1

    vidObj.release()
    cv2.destroyAllWindows()
    gc.collect()
    return frames_saved

# Function to classify frames
def classify_frames(frame_paths, model, transform):
    results = []
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        frame = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = frame.unsqueeze(0)  # Add a batch dimension
        output = model(frame)
        _, predicted = torch.max(output.data, 1)
        results.append(predicted.item())
    return results

# Use Streamlit's caching to only load the model once
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    model = resnext50_32x4d(pretrained=False)  # Define the model architecture
    state_dict = torch.load('ResNeXt_best_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)  # Load the state dictionary
    return model

# Set up transformations
@st.cache
def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

model = load_model()
transform = get_transforms()

# Main function to run the Streamlit app
def main():
    st.title("Deepfake Detection")

    # Upload video file
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    # Directory to save frames
    frames_dir = './temp_frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    if uploaded_video is not None:
        if st.button("Process Video"):
            # Save frames from the video
            with st.spinner('Processing video into frames...'):
                video_path = os.path.join(frames_dir, "uploaded_video.mp4")
                with open(video_path, "wb") as f:
                    f.write(uploaded_video.read())
                num_frames = save_frames_from_video(video_path, frames_dir)

            # Classify the frames
            frame_paths = glob.glob(os.path.join(frames_dir, '*.jpg'))
            with st.spinner('Analyzing frames...'):
                results = classify_frames(frame_paths, model, transform)

            # Show the results
            st.write("Results:")
            for i, result in enumerate(results):
                st.write(f"Frame {i+1}: Class {result}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
