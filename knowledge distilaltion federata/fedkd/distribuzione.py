import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
import os, json

# === Parametri e path ===
user_path1 = 'D:/tesi/infocom_poster-main/progetto/femnist_lab/'
user_path2 = 'D:/tesi/infocom_poster-main/infocom_poster-main/irds_dataset/'
ds = 'irds'
num_classes = 4
batch_size = 64

# ========== DATA LOADING (unchanged) ==========
def load_data(partition_id: int, num_partitions: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Load federated data partitions"""
    if ds == 'femnist':
        base_path = os.path.join(user_path1, "femnist_processed", f"client_{partition_id}")
        
        train_feat_path = os.path.join(base_path, "train_features.pt")
        train_label_path = os.path.join(base_path, "train_labels.pt")
        test_feat_path = os.path.join(base_path, "test_features.pt")
        test_label_path = os.path.join(base_path, "test_labels.pt")
        
        for path in [train_feat_path, train_label_path, test_feat_path, test_label_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")
        
        train_features = torch.load(train_feat_path)
        train_labels = torch.load(train_label_path)
        test_features = torch.load(test_feat_path)
        test_labels = torch.load(test_label_path)
        
        train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    else:  # irds dataset
        map_dirs = {
            '0': '101', '1': '102', '2': '103', 
            '3': '104', '4': '105', '5': '106', '6': '107'
        }

        path = f'{user_path2}/{map_dirs[str(partition_id)]}'

        train_dir_len = len(os.listdir(f'{path}/train'))
        validation_dir_len = len(os.listdir(f'{path}/validation'))
        test_dir_len = len(os.listdir(f'{path}/test'))
        
        train_map_file = json.load(open(f'{path}/quantized_train_files_map.json'))
        validation_map_file = json.load(open(f'{path}/quantized_validation_files_map.json'))
        test_map_file = json.load(open(f'{path}/quantized_test_files_map.json'))

        # Load training data
        train_labels = np.array([])
        for i in range(train_dir_len):
            train_data_tensor = torch.load(f'{path}/train/{i}_{map_dirs[str(partition_id)]}_train_quantized_features.pt')
            train_labels = np.append(train_labels, int(train_map_file[f'{i}_{map_dirs[str(partition_id)]}_train_quantized_features.pt']))
            if i == 0:
                train_data = train_data_tensor
            else:
                train_data = torch.cat((train_data, train_data_tensor), 0)
        
        # Load validation data
        validation_labels = np.array([])
        for i in range(validation_dir_len):
            validation_data_tensor = torch.load(f'{path}/validation/{i}_{map_dirs[str(partition_id)]}_validation_quantized_features.pt')
            validation_labels = np.append(validation_labels, int(validation_map_file[f'{i}_{map_dirs[str(partition_id)]}_validation_quantized_features.pt']))
            if i == 0:
                validation_data = validation_data_tensor
            else:
                validation_data = torch.cat((validation_data, validation_data_tensor), 0)
        
        # Load test data
        test_labels = np.array([])
        for i in range(test_dir_len):
            test_data_tensor = torch.load(f'{path}/test/{i}_{map_dirs[str(partition_id)]}_test_quantized_features.pt')
            test_labels = np.append(test_labels, int(test_map_file[f'{i}_{map_dirs[str(partition_id)]}_test_quantized_features.pt']))
            if i == 0:
                test_data = test_data_tensor
            else:
                test_data = torch.cat((test_data, test_data_tensor), 0)
        
        # Combine train and validation for training
        combined_train_data = torch.cat((train_data, validation_data), 0)
        combined_train_labels = np.append(train_labels, validation_labels)
        
        train_loader = DataLoader(
            TensorDataset(combined_train_data, torch.tensor(combined_train_labels, dtype=torch.long)),
            batch_size=batch_size, shuffle=True
        )
        
        test_loader = DataLoader(
            TensorDataset(test_data, torch.tensor(test_labels, dtype=torch.long)),
            batch_size=batch_size, shuffle=False
        )
        
        return train_loader, test_loader
# === Calcolo distribuzione classi per ogni client ===
num_clients = 7
class_distribution = {}

for cid in range(num_clients):
    train_loader, _ = load_data(cid, num_clients, batch_size)
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())
    counts = Counter(all_labels)
    class_distribution[cid] = [counts.get(cls, 0) for cls in range(num_classes)]

# === Plot istogramma migliorato con legenda sotto ===
x = np.arange(num_clients)
width = 0.1
colors = plt.get_cmap("tab10").colors

plt.style.use('seaborn-v0_8-muted')
fig, ax = plt.subplots(figsize=(12, 7))

for i in range(num_classes):
    values = [class_distribution[cid][i] for cid in range(num_clients)]
    ax.bar(x + i * width, values, width, label=f'Classe {i}', color=colors[i])

# Etichette e titolo
ax.set_xlabel('Client (Paziente)', fontsize=16, labelpad=10)
ax.set_ylabel('Numero di esempi', fontsize=16, labelpad=10)
ax.set_title('Distribuzione delle classi per client – IRDS', fontsize=18, weight='bold')

# Assi e ticks
ax.set_xticks(x + width * (num_classes - 1) / 2)
ax.set_xticklabels([f'C{cid}' for cid in range(num_clients)], fontsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.tick_params(axis='x', labelsize=13)

# Griglia
ax.grid(axis='y', linestyle='--', alpha=0.5)

# === Legenda centrata sotto ===
legend = ax.legend(
    title="Classi", fontsize=12, title_fontsize=13,
    ncol=5, loc='upper center',
    bbox_to_anchor=(0.5, -0.25),  # più spazio sotto
    frameon=False
)

# === Margini e salvataggio ===
plt.tight_layout()
plt.subplots_adjust(bottom=0.30)  # aumenta lo spazio per farci stare la legenda
plt.savefig("femnist_class_distribution_clean.png", dpi=300, bbox_inches='tight')
plt.show()
