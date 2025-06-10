# train_release_model.py

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "release_patches"  # The base directory where your 'video_name/shot_XXX' folders are
LABEL_FILE = "release_patches/GH012211/labels.csv"  # The CSV file with your manual labels
PATCH_SIZE = 96  # Must match the size from data_collection.py
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
NUM_EPOCHS = 20
SEQUENCE_LENGTH = 2  # Using pairs of frames (t and t-1)

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# --- 1. PyTorch Dataset Class ---
class ReleaseDataset(Dataset):
    def __init__(self, data_samples, transform=None):
        """
        Args:
            data_samples (list of dicts): List containing sample info.
                                         Each dict: {'hand_t', 'hand_t-1', 'ball_t', 'ball_t-1', 'label'}
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_samples = data_samples
        self.transform = transform

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        # Load all four images
        hand_t = cv2.imread(sample['hand_t'])
        hand_t_minus_1 = cv2.imread(sample['hand_t-1'])
        ball_t = cv2.imread(sample['ball_t'])
        ball_t_minus_1 = cv2.imread(sample['ball_t-1'])

        # Convert BGR to RGB
        hand_t = cv2.cvtColor(hand_t, cv2.COLOR_BGR2RGB)
        hand_t_minus_1 = cv2.cvtColor(hand_t_minus_1, cv2.COLOR_BGR2RGB)
        ball_t = cv2.cvtColor(ball_t, cv2.COLOR_BGR2RGB)
        ball_t_minus_1 = cv2.cvtColor(ball_t_minus_1, cv2.COLOR_BGR2RGB)

        label = torch.tensor([sample['label']], dtype=torch.float32)

        if self.transform:
            hand_t = self.transform(hand_t)
            hand_t_minus_1 = self.transform(hand_t_minus_1)
            ball_t = self.transform(ball_t)
            ball_t_minus_1 = self.transform(ball_t_minus_1)

        return hand_t, hand_t_minus_1, ball_t, ball_t_minus_1, label


# --- 2. Model Architecture ---
class FeatureExtractor(nn.Module):
    """A simple CNN to extract features from a sequence of 2 patches."""

    def __init__(self, in_channels=6, out_features=128):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 96x96 -> 48x48
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 48x48 -> 24x24
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 24x24 -> 12x12
            nn.BatchNorm2d(128),

            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten()
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x_t, x_t_minus_1):
        # Concatenate along the channel dimension
        x = torch.cat((x_t, x_t_minus_1), dim=1)
        features = self.conv_net(x)
        return self.fc(features)


class ReleaseRelationNet(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        # Use a single shared-weight feature extractor
        self.feature_extractor = FeatureExtractor(out_features=feature_dim)

        self.relation_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, hand_t, hand_t_minus_1, ball_t, ball_t_minus_1):
        hand_features = self.feature_extractor(hand_t, hand_t_minus_1)
        ball_features = self.feature_extractor(ball_t, ball_t_minus_1)

        combined_features = torch.cat((hand_features, ball_features), dim=1)
        release_score = self.relation_head(combined_features)
        return release_score


# --- 3. Data Preparation ---
def prepare_data(data_dir, label_file):
    print("Preparing data...")
    try:
        labels_df = pd.read_csv(label_file)
    except FileNotFoundError:
        print(f"Error: Label file not found at '{label_file}'")
        return []
    all_samples = []
    for _, row in labels_df.iterrows():
        video_name = row['video_name']
        shot_id = row['shot_id']
        true_release_frame = row['true_release_frame']

        shot_folder = os.path.join(data_dir, video_name, f"shot_{shot_id:03d}")

        if not os.path.isdir(shot_folder):
            print(f"Warning: Shot folder not found: {shot_folder}")
            continue

        # Get all frame numbers available for this shot
        frame_files = os.listdir(shot_folder)
        available_frames = sorted(list(set([int(f.split('_')[1]) for f in frame_files if f.startswith('frame_')])))

        if not available_frames:
            print(f"Warning: No valid frame files found in {shot_folder}")
            continue

        # --- Handle Labeling ---
        if true_release_frame == -1:
            # This is a "hard negative" shot. All frames get a label of 0.0
            label_map = {f_num: 0.0 for f_num in available_frames}
        else:
            # This is a good shot, labels depend on the frame number
            label_map = {f_num: (1.0 if f_num >= true_release_frame else 0.0) for f_num in available_frames}

        # --- Create Samples by Finding Previous Available Frame ---
        for i, f_num_t in enumerate(available_frames):
            # We need a previous frame, so we can't create a sample for the very first frame in our list.
            if i == 0:
                continue

            # The previous frame is simply the one at index i-1 in our sorted list.
            f_num_t_minus_1 = available_frames[i - 1]

            # Now we construct the paths. We know these files exist because we built
            # `available_frames` from the file list.
            hand_t_path = os.path.join(shot_folder, f"frame_{f_num_t:05d}_hand.png")
            hand_t_minus_1_path = os.path.join(shot_folder, f"frame_{f_num_t_minus_1:05d}_hand.png")
            ball_t_path = os.path.join(shot_folder, f"frame_{f_num_t:05d}_ball.png")
            ball_t_minus_1_path = os.path.join(shot_folder, f"frame_{f_num_t_minus_1:05d}_ball.png")

            # This check is now slightly redundant but safe. It mainly ensures both _hand and _ball patches exist for a given frame number.
            if os.path.exists(hand_t_path) and os.path.exists(ball_t_path) and \
                    os.path.exists(hand_t_minus_1_path) and os.path.exists(ball_t_minus_1_path):

                # Get the label for frame `f_num_t`
                label = label_map.get(f_num_t)  # Use .get for safety, though it should exist

                if label is not None:
                    all_samples.append({
                        'hand_t': hand_t_path,
                        'hand_t-1': hand_t_minus_1_path,
                        'ball_t': ball_t_path,
                        'ball_t-1': ball_t_minus_1_path,
                        'label': label
                    })
    print(f"Total samples created: {len(all_samples)}")
    return all_samples


# --- 4. Training and Validation Loop ---
def train_model():
    # Prepare data
    all_samples = prepare_data(DATA_DIR, LABEL_FILE)
    if not all_samples:
        print("No data samples found. Exiting.")
        return

    # Split data: 80% train, 20% validation
    train_samples, val_samples = train_test_split(all_samples, test_size=0.2, random_state=42, shuffle=True)
    print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    # Define transforms
    # For training, add augmentation. For validation, just ToTensor and Normalize.
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create datasets and dataloaders
    train_dataset = ReleaseDataset(train_samples, transform=train_transform)
    val_dataset = ReleaseDataset(val_samples, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, optimizer, criterion
    model = ReleaseRelationNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()  # Binary Cross-Entropy for 0-1 scores

    # --- Training Loop ---
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for hand_t, hand_t_minus_1, ball_t, ball_t_minus_1, labels in train_loader:
            # Move to device
            hand_t, hand_t_minus_1 = hand_t.to(DEVICE), hand_t_minus_1.to(DEVICE)
            ball_t, ball_t_minus_1 = ball_t.to(DEVICE), ball_t_minus_1.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(hand_t, hand_t_minus_1, ball_t, ball_t_minus_1)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * hand_t.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for hand_t, hand_t_minus_1, ball_t, ball_t_minus_1, labels in val_loader:
                hand_t, hand_t_minus_1 = hand_t.to(DEVICE), hand_t_minus_1.to(DEVICE)
                ball_t, ball_t_minus_1 = ball_t.to(DEVICE), ball_t_minus_1.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(hand_t, hand_t_minus_1, ball_t, ball_t_minus_1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * hand_t.size(0)

                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = 100 * correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} -> Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_release_model.pth')
            print(f"   -> Model saved with new best validation loss: {best_val_loss:.4f}")

    return history


def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Validation Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


if __name__ == '__main__':
    training_history = train_model()
    if training_history:
        plot_training_history(training_history)