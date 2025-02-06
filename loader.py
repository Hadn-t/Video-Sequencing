import os
import json
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WLASLDataset(Dataset):
    def __init__(self, json_path, videos_path, num_frames=16, transform=None):
        """
        Args:
            json_path (str): Path to WLASL_v0.3.json.
            videos_path (str): Path to the folder containing video files.
            num_frames (int): Number of frames to sample from each video.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.videos_path = videos_path
        self.num_frames = num_frames
        self.transform = transform

        # Load the JSON metadata
        with open(json_path, 'r', encoding="utf-8") as f:
            self.data = json.load(f)  # ✅ FIXED (load list instead of expecting a dictionary)

        # Extract all valid video entries
        self.video_list = []
        self.labels = []  # Stores class labels
        self.label_map = {}  # Mapping ASL words to index

        for idx, entry in enumerate(self.data):
            word = entry["gloss"]  # ASL word label
            if word not in self.label_map:
                self.label_map[word] = len(self.label_map)  # Assign unique index

            if "instances" in entry:
                for instance in entry["instances"]:
                    video_id = instance.get("video_id", None)
                    if video_id:
                        video_path = os.path.join(videos_path, f"{video_id}.mp4")
                        if os.path.exists(video_path):
                            self.video_list.append(video_path)
                            self.labels.append(self.label_map[word])

    def __len__(self):
        return len(self.video_list)

    def load_video_frames(self, video_path):
        """Loads a fixed number of frames from a video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idxs = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # If video has fewer frames than required, pad with last frame
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return np.array(frames)

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        label = self.labels[idx]

        frames = self.load_video_frames(video_path)

        # Apply transformations if any
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])

        return frames, torch.tensor(label, dtype=torch.long)


# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset path (update accordingly)
json_path = r"E:\archive\WLASL_v0.3.json"
videos_path = r"E:\archive\videos"

if __name__ == '__main__':
    # Initialize dataset
    dataset = WLASLDataset(json_path, videos_path, transform=transform)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # Test
    sample_frames, sample_label = next(iter(dataloader))
    print("Batch shape:", sample_frames.shape)  # Expected: (8, 16, 3, 112, 112)
    print("Labels:", sample_label)
