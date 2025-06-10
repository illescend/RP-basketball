# model_def.py

import torch
import torch.nn as nn

# This file contains the model architecture definition so it can be shared
# between training, testing, and inference scripts.

class FeatureExtractor(nn.Module):
    """A simple CNN to extract features from a sequence of 2 patches."""
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