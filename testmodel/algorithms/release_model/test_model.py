# test_model.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "release_patches"  # The base directory where your 'video_name/shot_XXX' folders are
LABEL_FILE = "release_patches/GH012211/labels.csv"  # The CSV file with your manual labels
MODEL_PATH = "best_release_model.pth"  # Path to your trained model
PATCH_SIZE = 96
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SHOTS_TO_TEST = 5  # How many random shots to visualize


# --- Copy Model and Dataset Class Definitions from train_release_model.py ---
# (It's better to put these in a shared model_def.py file and import them,
# but copying is fine for a self-contained test script)

class ReleaseDataset(Dataset):
    def __init__(self, data_samples, transform=None):
        self.data_samples = data_samples
        self.transform = transform

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        hand_t = cv2.cvtColor(cv2.imread(sample['hand_t']), cv2.COLOR_BGR2RGB)
        hand_t_minus_1 = cv2.cvtColor(cv2.imread(sample['hand_t-1']), cv2.COLOR_BGR2RGB)
        ball_t = cv2.cvtColor(cv2.imread(sample['ball_t']), cv2.COLOR_BGR2RGB)
        ball_t_minus_1 = cv2.cvtColor(cv2.imread(sample['ball_t-1']), cv2.COLOR_BGR2RGB)
        label = torch.tensor([sample['label']], dtype=torch.float32)

        if self.transform:
            hand_t = self.transform(hand_t)
            hand_t_minus_1 = self.transform(hand_t_minus_1)
            ball_t = self.transform(ball_t)
            ball_t_minus_1 = self.transform(ball_t_minus_1)

        # Also return the frame number for plotting
        frame_num_t = int(os.path.basename(sample['hand_t']).split('_')[1])

        return hand_t, hand_t_minus_1, ball_t, ball_t_minus_1, label, frame_num_t


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=6, out_features=128):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x_t, x_t_minus_1):
        x = torch.cat((x_t, x_t_minus_1), dim=1)
        features = self.conv_net(x)
        return self.fc(features)


class ReleaseRelationNet(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_extractor = FeatureExtractor(out_features=feature_dim)
        self.relation_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, hand_t, hand_t_minus_1, ball_t, ball_t_minus_1):
        hand_features = self.feature_extractor(hand_t, hand_t_minus_1)
        ball_features = self.feature_extractor(ball_t, ball_t_minus_1)
        combined_features = torch.cat((hand_features, ball_features), dim=1)
        return self.relation_head(combined_features)


# --- Main Inference and Visualization Function ---
def test_and_visualize():
    # 1. Load the trained model
    print(f"Loading model from {MODEL_PATH}")
    model = ReleaseRelationNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please train the model first.")
        return
    model.eval()  # Set the model to evaluation mode

    # 2. Load the label manifest to find shots to test
    try:
        labels_df = pd.read_csv(LABEL_FILE)
    except FileNotFoundError:
        print(f"Error: Label file not found at '{LABEL_FILE}'")
        return

    # Get a list of unique shots (video_name, shot_id)
    unique_shots = labels_df.drop_duplicates(subset=['video_name', 'shot_id'])

    # 3. Randomly select shots to test
    num_available_shots = len(unique_shots)
    if num_available_shots == 0:
        print("No shots found in label file.")
        return

    num_to_select = min(NUM_SHOTS_TO_TEST, num_available_shots)
    selected_shots = unique_shots.sample(n=num_to_select)
    print(f"\nRandomly selected {num_to_select} shots to test...")

    # Define validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 4. Loop through selected shots and perform inference
    for _, shot_info in selected_shots.iterrows():
        video_name = shot_info['video_name']
        shot_id = shot_info['shot_id']
        true_release_frame = shot_info['true_release_frame']

        print(f"\n--- Testing Shot: {video_name} / shot_{shot_id:03d} ---")

        # Prepare data just for this one shot
        shot_folder = os.path.join(DATA_DIR, video_name, f"shot_{shot_id:03d}")
        if not os.path.isdir(shot_folder):
            print(f"  -> Shot folder not found. Skipping.")
            continue

        available_frames = sorted(
            list(set([int(f.split('_')[1]) for f in os.listdir(shot_folder) if f.startswith('frame_')])))

        shot_samples = []
        for i, f_num_t in enumerate(available_frames):
            if i == 0: continue
            f_num_t_minus_1 = available_frames[i - 1]

            hand_t_path = os.path.join(shot_folder, f"frame_{f_num_t:05d}_hand.png")
            ball_t_path = os.path.join(shot_folder, f"frame_{f_num_t:05d}_ball.png")
            hand_t_minus_1_path = os.path.join(shot_folder, f"frame_{f_num_t_minus_1:05d}_hand.png")
            ball_t_minus_1_path = os.path.join(shot_folder, f"frame_{f_num_t_minus_1:05d}_ball.png")

            if os.path.exists(hand_t_path) and os.path.exists(ball_t_path) and os.path.exists(
                    hand_t_minus_1_path) and os.path.exists(ball_t_minus_1_path):
                label = 1.0 if (true_release_frame != -1 and f_num_t >= true_release_frame) else 0.0
                shot_samples.append({'hand_t': hand_t_path, 'hand_t-1': hand_t_minus_1_path,
                                     'ball_t': ball_t_path, 'ball_t-1': ball_t_minus_1_path,
                                     'label': label})

        if not shot_samples:
            print("  -> No valid samples found for this shot. Skipping.")
            continue

        # Create dataset and dataloader for this single shot
        shot_dataset = ReleaseDataset(shot_samples, transform=val_transform)
        shot_loader = DataLoader(shot_dataset, batch_size=8, shuffle=False)

        # Run inference
        all_preds = []
        all_labels = []
        all_frame_nums = []
        with torch.no_grad():
            for hand_t, hand_t_minus_1, ball_t, ball_t_minus_1, labels, frame_nums in shot_loader:
                hand_t, hand_t_minus_1 = hand_t.to(DEVICE), hand_t_minus_1.to(DEVICE)
                ball_t, ball_t_minus_1 = ball_t.to(DEVICE), ball_t_minus_1.to(DEVICE)

                outputs = model(hand_t, hand_t_minus_1, ball_t, ball_t_minus_1)

                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.numpy().flatten())
                all_frame_nums.extend(frame_nums.numpy())

        # 5. Plot the results for this shot
        plt.figure(figsize=(15, 6))

        # Sort by frame number for clean plotting
        sorted_indices = np.argsort(all_frame_nums)
        sorted_frames = np.array(all_frame_nums)[sorted_indices]
        sorted_preds = np.array(all_preds)[sorted_indices]
        sorted_labels = np.array(all_labels)[sorted_indices]

        plt.plot(sorted_frames, sorted_labels, 'go--', label='Ground Truth Label', marker='o')
        plt.plot(sorted_frames, sorted_preds, 'r-', label='Model Prediction Score', marker='x')
        plt.axvline(x=true_release_frame, color='b', linestyle=':', label=f'True Release Frame ({true_release_frame})')
        plt.title(f'Model Performance on: {video_name} / shot_{shot_id:03d}')
        plt.xlabel('Frame Number')
        plt.ylabel('Score (0 = No Release, 1 = Release)')
        plt.ylim(-0.1, 1.1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()

        plot_save_path = f"test_result_shot_{video_name}_{shot_id:03d}.png"
        plt.savefig(plot_save_path)
        print(f"  -> Saved plot to {plot_save_path}")
        plt.show()


if __name__ == '__main__':
    test_and_visualize()