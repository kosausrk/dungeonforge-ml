#I: 
"""
🧠 ML Dungeon Generator | CS4780 Project

Project Purpose:
This project applies machine learning to generate structured dungeon maps procedurally.
We simulate dungeon grid layouts and learn to cluster and generate new, playable maps using
unsupervised learning (K-Means) and generative modeling (Variational Autoencoder).

Why It Fits CS4780 (Machine Learning for Intelligent Systems):
- Uses **unsupervised clustering** to analyze level design structure (K-Means)
- Implements a **generative model (VAE)** to learn and sample new layouts
- Applies **probabilistic reasoning** through latent variable modeling in the VAE
- Includes **evaluation** via visual inspection of novelty, structure, and interpretability

🧱 Domain Breakdown / UML-Style Structure:
[DungeonGridGenerator] → generates path-based layouts (entry/exit)
[ClusterAnalyzer] → KMeans finds latent clusters in map design
[DungeonVAE] → learns compressed representation and regenerates samples
[Visualizer] → shows output maps in a meaningful, color-coded way
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# 1. Grid-Based Dungeon Layout Generator
def generate_dungeons(n=1000, size=10):
    """
    Generate dungeon layouts with a path from top-left (entry) to bottom-right (exit).
    """
    dungeons = []
    for _ in range(n):
        grid = np.zeros((size, size), dtype=int)

        x, y = 0, 0
        grid[x, y] = 2  # Entry

        while x < size - 1 or y < size - 1:
            if np.random.rand() < 0.5 and x < size - 1:
                x += 1
            elif y < size - 1:
                y += 1
            grid[x, y] = 1  # Path

        grid[size - 1, size - 1] = 3  # Exit

        # Add noise (more path tiles)
        noise = np.random.rand(size, size) < 0.1
        grid[(grid == 0) & noise] = 1

        dungeons.append(grid.flatten())
    return np.array(dungeons)


# 2. Clustering Dungeon Styles
def cluster_dungeons(data, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans


# 3. VAE Model Definition
class VAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=10):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc21 = nn.Linear(64, latent_dim)
        self.fc22 = nn.Linear(64, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# 4. VAE Loss & Training
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(model, data, epochs=20, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x in data:
            x = torch.tensor(x, dtype=torch.float32)

            # ✅ Clip input to [0, 1] so entry/exit (2/3) become 1 for training
            x_clipped = torch.clamp(x, 0, 1)

            recon_x, mu, logvar = model(x_clipped)
            loss = loss_function(recon_x, x_clipped, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

# 5. Visualize Sampled Dungeons
def sample_dungeons(model, n=5, latent_dim=10, grid_size=10):
    import matplotlib.colors as mcolors

    model.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim)
        samples = model.decode(z).numpy().reshape(n, grid_size, grid_size)

        # Color mapping: 0=Wall, 1=Path, 2=Entry, 3=Exit
        tile_colors = {
            0: "#2c3e50",  # Wall - dark blue-gray
            1: "#ecf0f1",  # Path - light
            2: "#27ae60",  # Entry - green
            3: "#c0392b"   # Exit - red
        }

        cmap = mcolors.ListedColormap([tile_colors[i] for i in range(4)])
        bounds = [0, 1, 2, 3, 4]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        for i, sample in enumerate(samples):
            dungeon = (sample > 0.5).astype(int)
            dungeon[0, 0] = 2  # entry
            dungeon[-1, -1] = 3  # exit

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(dungeon, cmap=cmap, norm=norm)

            # Add gridlines
            ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.grid(which="minor", color="black", linewidth=0.5)
            ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

            ax.set_title(f"Generated Dungeon {i+1}", fontsize=14)
            plt.tight_layout()
            plt.show()



#trying to use BFS search 

from collections import deque

def simulate_player(dungeon, entry_val=2, exit_val=3):
    """
    Simulates a player navigating from entry (2) to exit (3).
    Uses BFS to determine if a path exists.
    """
    size = dungeon.shape[0]
    visited = np.zeros_like(dungeon, dtype=bool)

    # Find start and end positions
    start = tuple(np.argwhere(dungeon == entry_val)[0])
    goal = tuple(np.argwhere(dungeon == exit_val)[0])

    # BFS queue
    queue = deque([start])
    visited[start] = True

    # Movement: up, down, left, right
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True  # Reached exit

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                if not visited[nx, ny] and dungeon[nx, ny] in [1, 3]:
                    visited[nx, ny] = True
                    queue.append((nx, ny))

    return False  # No path found


def simulate_player(dungeon, entry_val=2, exit_val=3):
    """
    Simulates a player navigating from entry (2) to exit (3).
    Uses BFS to determine if a valid path exists.
    """
    size = dungeon.shape[0]
    visited = np.zeros_like(dungeon, dtype=bool)

    # Find coordinates of entry and exit
    try:
        start = tuple(np.argwhere(dungeon == entry_val)[0])
        goal = tuple(np.argwhere(dungeon == exit_val)[0])
    except IndexError:
        return False  # Entry or exit missing

    queue = deque([start])
    visited[start] = True
    directions = [(-1,0), (1,0), (0,-1), (0,1)]  # 4-way movement

    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                if not visited[nx, ny] and dungeon[nx, ny] in [1, 3]:
                    visited[nx, ny] = True
                    queue.append((nx, ny))

    return False  # No valid path

def evaluate_dungeon(model, latent_dim=10, grid_size=10, trials=10):
    """
    Generates `trials` number of dungeons from the model and evaluates
    how many are actually playable (entry to exit reachable).
    """
    model.eval()
    reachable_count = 0

    with torch.no_grad():
        z = torch.randn(trials, latent_dim)
        samples = model.decode(z).numpy().reshape(trials, grid_size, grid_size)

        for i, sample in enumerate(samples):
            dungeon = (sample > 0.5).astype(int)
            dungeon[0, 0] = 2  # entry
            dungeon[-1, -1] = 3  # exit

            if simulate_player(dungeon):
                reachable_count += 1

    print(f"✅ {reachable_count}/{trials} generated dungeons are playable.")



# 6. Run Main
if __name__ == "__main__":
    print("🧱 Generating structured dungeon data...")
    data = generate_dungeons()
    
    print("🔍 Clustering dungeon layout types...")
    labels, _ = cluster_dungeons(data)

    print("🧠 Training Variational Autoencoder (VAE)...")
    vae = VAE(input_dim=100)
    train_vae(vae, data)

    print("🎲 Sampling new dungeons from latent space...")
    sample_dungeons(vae)


#II: 
"""
🎮 ML Dungeon Generator — CS4780 Project

This project uses unsupervised learning (KMeans) and a generative model (VAE) to create new, playable dungeon maps. 
It starts by generating random but structured dungeon grids with entry and exit points. KMeans clusters layouts 
to find common patterns, while the Variational Autoencoder learns to compress and recreate them. Over multiple 
epochs (training cycles), the model improves its ability to generate realistic dungeons from random inputs. 
This applies core CS4780 concepts like clustering, dimensionality reduction, probabilistic modeling, and evaluation 
of generated outputs. Practical use cases include automated level design and AI-based map generation in games.
"""


#Practicaliyy 
# Game Level Generation (Indie/AAA Devs)
# Use this system to auto-generate endless dungeon maps in games like Dead Cells, Hades, or Binding of Isaac.





