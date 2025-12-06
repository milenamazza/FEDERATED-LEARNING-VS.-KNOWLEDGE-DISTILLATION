"""
Knowledge Distillation System with Standard Neural Networks
Enhanced with cross-client matrix CSV logging (no image plots)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import copy
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import random
import itertools
from datetime import datetime

def fix_all_seeds(seed: int = 42):
    """Fix all random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# =============================================================================
# STANDARD NEURAL NETWORK ARCHITECTURES
# =============================================================================

class TeacherNetwork(nn.Module):
    """Standard deep neural network for teacher model"""
    
    def __init__(self, input_size=1280, hidden_sizes=[512, 256, 128], num_classes=4, 
                 dropout_rate=0.3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(TeacherNetwork, self).__init__()

        self.device = device
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Final classification layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class StudentNetwork(nn.Module):
    """Lightweight neural network for student model"""
    
    def __init__(self, input_size=1280, hidden_sizes=[256, 128], num_classes=4, 
                 dropout_rate=0.2, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(StudentNetwork, self).__init__()

        self.device = device
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Final classification layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def standard_classification_loss(outputs, labels):
    """Standard cross-entropy loss for classification"""
    return F.cross_entropy(outputs, labels.long())

def knowledge_distillation_loss(student_outputs, teacher_outputs, labels, 
                              temperature=4.0, alpha=0.7):
    """Knowledge Distillation Loss combining soft and hard targets"""
    
    # Hard target loss (standard classification loss)
    hard_loss = standard_classification_loss(student_outputs, labels)
    
    # Soft target loss (knowledge distillation)
    teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)
    student_log_probs = F.log_softmax(student_outputs / temperature, dim=1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    soft_loss *= (temperature ** 2)
    
    # Combined loss
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return total_loss, soft_loss, hard_loss

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_validation_data_with_split(user_path='irds_dataset', batch_size=32, num_clients=7, 
                                  teacher_val_ratio=0.2):
    """Load validation data and split into teacher validation and remaining data"""
    
    map_dirs = {
        '0': '101', '1': '102', '2': '103', '3': '104',
        '4': '105', '5': '106', '6': '107'
    }
    
    all_validation_data = []
    all_validation_labels = []
    
    for partition_id in range(num_clients):
        path = f'{user_path}/{map_dirs[str(partition_id)]}'
        
        validation_dir_len = len(os.listdir(f'{path}/validation'))
        validation_map_file = json.load(open(f'{path}/quantized_validation_files_map.json'))
        
        for i in range(validation_dir_len):
            validation_data_tensor = torch.load(f'{path}/validation/{i}_{map_dirs[str(partition_id)]}_validation_quantized_features.pt')
            label = int(validation_map_file[f'{i}_{map_dirs[str(partition_id)]}_validation_quantized_features.pt'])
            
            all_validation_data.append(validation_data_tensor)
            all_validation_labels.extend([label] * validation_data_tensor.shape[0])
    
    # Combine all validation data
    combined_data = torch.cat(all_validation_data, dim=0)
    combined_labels = torch.tensor(all_validation_labels, dtype=torch.float32)

    # Invece di random_split, usa stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        combined_data, combined_labels, 
        test_size=teacher_val_ratio, 
        stratify=combined_labels,  # Mantiene proporzioni classi
        random_state=42
    )
    
    # Create dataset and split
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    teacher_val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    teacher_val_loader = torch.utils.data.DataLoader(
        teacher_val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, teacher_val_loader

def load_client_data(partition_id: int, user_path='irds_dataset', batch_size=32):
    """Load train+test data for a specific client"""
    
    map_dirs = {
        '0': '101', '1': '102', '2': '103', '3': '104',
        '4': '105', '5': '106', '6': '107'
    }
    
    path = f'{user_path}/{map_dirs[str(partition_id)]}'
    
    # Load train data
    train_dir_len = len(os.listdir(f'{path}/train'))
    train_map_file = json.load(open(f'{path}/quantized_train_files_map.json'))
    
    train_data_list = []
    train_labels = []
    
    for i in range(train_dir_len):
        train_data_tensor = torch.load(f'{path}/train/{i}_{map_dirs[str(partition_id)]}_train_quantized_features.pt')
        label = int(train_map_file[f'{i}_{map_dirs[str(partition_id)]}_train_quantized_features.pt'])
        train_data_list.append(train_data_tensor)
        train_labels.extend([label] * train_data_tensor.shape[0])
    
    # Load test data
    test_dir_len = len(os.listdir(f'{path}/test'))
    test_map_file = json.load(open(f'{path}/quantized_test_files_map.json'))
    
    test_data_list = []
    test_labels = []
    
    for i in range(test_dir_len):
        test_data_tensor = torch.load(f'{path}/test/{i}_{map_dirs[str(partition_id)]}_test_quantized_features.pt')
        label = int(test_map_file[f'{i}_{map_dirs[str(partition_id)]}_test_quantized_features.pt'])
        test_data_list.append(test_data_tensor)
        test_labels.extend([label] * test_data_tensor.shape[0])
    
    # Combine data
    train_data = torch.cat(train_data_list, dim=0)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    test_data = torch.cat(test_data_list, dim=0)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_labels_tensor),
        batch_size=batch_size, shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data, test_labels_tensor),
        batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader


def load_all_client_data(user_path='femnist_dataset', batch_size=32, num_clients=7):
    """Load data for all clients and return as dictionary"""
    
    all_client_data = {}
    
    for client_id in range(num_clients):
        train_loader, val_loader = load_client_data(client_id, user_path, batch_size)
        all_client_data[client_id] = {
            'train_loader': train_loader,
            'val_loader': val_loader
        }
    
    return all_client_data

# =============================================================================
# TRAINING FUNCTIONS WITH BEST MODEL SELECTION
# =============================================================================

def train_teacher_with_best_model_selection(teacher_model, train_loader, val_loader, device, 
                                          epochs=50, lr=0.001, gamma=0.95, step=15):
    """Train teacher model and save the best one based on validation accuracy"""
    
    print("Training Teacher Model...")
    teacher_model.to(device)
    
    optimizer = torch.optim.AdamW(
        teacher_model.parameters(), 
        lr=lr,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    
    best_accuracy = 0.0
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_loss = 0
    best_model_state = None
    teacher_history = []
    
    for epoch in range(epochs):
        # Training phase
        teacher_model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = teacher_model(features)
            
            loss = standard_classification_loss(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        train_accuracy = correct / total
        
        # Validation phase every epoch
        val_loss, val_accuracy, val_f1, val_precision, val_recall = evaluate_model(
            teacher_model, val_loader, device
        )
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_f1 = val_f1
            best_precision = val_precision
            best_recall = val_recall
            best_loss = val_loss
            best_model_state = copy.deepcopy(teacher_model.state_dict())
        
        # Record history
        teacher_history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss / len(train_loader),
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall
        })
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Teacher Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {epoch_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f} (Best: {best_accuracy:.4f})")
    
    # Load best model
    teacher_model.load_state_dict(best_model_state)
    
    # Get final metrics with best model
    final_metrics = evaluate_model(teacher_model, val_loader, device)
    
    teacher_results = {
        'best_accuracy': best_accuracy,
        'final_loss': best_loss,
        'best_accuracy': best_accuracy,
        'final_f1': best_f1,
        'final_precision': best_precision,
        'final_recall': best_recall,
        'training_history': teacher_history
    }
    
    print(f"Teacher training completed! Best validation accuracy: {best_accuracy:.4f}")
    return teacher_model, teacher_results

def train_student_with_best_model_selection(student_model, teacher_model, train_loader, val_loader, 
                                          device, epochs=30, lr=0.001, temperature=4.0, 
                                          alpha=0.7, gamma=0.9, step=10, client_id=0):
    """Train student model with knowledge distillation and save best model"""
    
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()
    
    optimizer = torch.optim.AdamW(
        student_model.parameters(), 
        lr=lr,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    
    best_accuracy = 0.0
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_loss = 0
    best_model_state = None
    student_history = []
    
    for epoch in range(epochs):
        # Training phase
        student_model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            student_outputs = student_model(features)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(features)
            
            total_loss, soft_loss, hard_loss = knowledge_distillation_loss(
                student_outputs, teacher_outputs, labels, temperature=temperature, alpha=alpha
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        train_accuracy = correct / total
        
        # Validation phase every epoch
        val_loss, val_accuracy, val_f1, val_precision, val_recall = evaluate_model(
            student_model, val_loader, device
        )
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_f1 = val_f1
            best_precision = val_precision
            best_recall = val_recall
            best_loss = val_loss
            best_model_state = copy.deepcopy(student_model.state_dict())
        
        # Record history
        student_history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss / len(train_loader),
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall
        })
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Client {client_id} Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {epoch_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f} (Best: {best_accuracy:.4f})")
    
    # Load best model
    student_model.load_state_dict(best_model_state)
    
    # Get final metrics with best model
    final_metrics = evaluate_model(student_model, val_loader, device)
    
    student_results = {
        'client_id': client_id,
        'best_accuracy': best_accuracy,
        'final_loss': best_loss,
        'best_accuracy': best_accuracy,
        'final_f1': best_f1,
        'final_precision': best_precision,
        'final_recall': best_recall,
        'training_history': student_history
    }
    
    return student_model, student_results

def evaluate_student_on_other_clients(student_model, client_id, all_client_data, device, num_clients):
    """Evaluate a student model on all other clients' test data"""
    
    other_client_results = {}
    cross_client_metrics = []
    
    print(f"  Evaluating Client {client_id} on other clients...")
    
    for other_client_id in range(num_clients):
        if other_client_id != client_id:  # Skip own client
            val_loader = all_client_data[other_client_id]['val_loader']
            
            # Evaluate on this client's data
            val_loss, val_accuracy, val_f1, val_precision, val_recall = evaluate_model(
                student_model, val_loader, device
            )
            
            other_client_results[f'cross_client_{other_client_id}'] = {
                'accuracy': val_accuracy,
                'f1': val_f1,
                'precision': val_precision,
                'recall': val_recall,
                'loss': val_loss
            }
            
            cross_client_metrics.append({
                'accuracy': val_accuracy,
                'f1': val_f1,
                'precision': val_precision,
                'recall': val_recall,
                'loss': val_loss
            })
    
    # Calculate average performance across other clients
    if cross_client_metrics:
        avg_cross_accuracy = np.mean([m['accuracy'] for m in cross_client_metrics])
        avg_cross_f1 = np.mean([m['f1'] for m in cross_client_metrics])
        avg_cross_precision = np.mean([m['precision'] for m in cross_client_metrics])
        avg_cross_recall = np.mean([m['recall'] for m in cross_client_metrics])
        avg_cross_loss = np.mean([m['loss'] for m in cross_client_metrics])
        
        std_cross_accuracy = np.std([m['accuracy'] for m in cross_client_metrics])
        
        print(f"    Client {client_id} avg cross-client accuracy: {avg_cross_accuracy:.4f} (±{std_cross_accuracy:.4f})")
        
        cross_client_summary = {
            'avg_cross_accuracy': avg_cross_accuracy,
            'avg_cross_f1': avg_cross_f1,
            'avg_cross_precision': avg_cross_precision,
            'avg_cross_recall': avg_cross_recall,
            'avg_cross_loss': avg_cross_loss,
            'std_cross_accuracy': std_cross_accuracy,
            'individual_results': other_client_results
        }
    else:
        cross_client_summary = {
            'avg_cross_accuracy': 0.0,
            'avg_cross_f1': 0.0,
            'avg_cross_precision': 0.0,
            'avg_cross_recall': 0.0,
            'avg_cross_loss': 0.0,
            'std_cross_accuracy': 0.0,
            'individual_results': {}
        }
    
    return cross_client_summary

def evaluate_model(model, test_loader, device):
    """Evaluate model performance"""
    
    model.to(device)
    model.eval()
    
    correct = 0
    total_loss = 0.0
    total_samples = 0
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = standard_classification_loss(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total_samples
    avg_loss = total_loss / len(test_loader)
    
    f1score = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1score, precision, recall

# =============================================================================
# CROSS-CLIENT MATRIX FUNCTIONS (NO PLOTTING)
# =============================================================================

def create_cross_client_matrix(client_results, num_clients, metric='accuracy'):
    """
    Create a matrix showing performance of each student model on each client's data
    
    Args:
        client_results: Results from all clients
        num_clients: Number of clients
        metric: Metric to display ('accuracy', 'f1', 'precision', 'recall')
    
    Returns:
        numpy array: Matrix where element [i,j] is performance of model i on client j's data
    """
    
    matrix = np.zeros((num_clients, num_clients))
    
    for client_id in range(num_clients):
        # Own client performance (diagonal)
        if metric == 'accuracy':
            matrix[client_id, client_id] = client_results[client_id]['best_accuracy']
        elif metric == 'f1':
            matrix[client_id, client_id] = client_results[client_id]['final_f1']
        elif metric == 'precision':
            matrix[client_id, client_id] = client_results[client_id]['final_precision']
        elif metric == 'recall':
            matrix[client_id, client_id] = client_results[client_id]['final_recall']
        
        # Cross-client performance
        individual_results = client_results[client_id]['cross_client_evaluation']['individual_results']
        for cross_client_key, cross_metrics in individual_results.items():
            target_client_id = int(cross_client_key.split('_')[-1])
            matrix[client_id, target_client_id] = cross_metrics[metric]
    
    return matrix

def print_cross_client_summary_table(matrix, num_clients, metric='accuracy'):
    """Print a nicely formatted cross-client performance table"""
    
    print(f"\n{'='*60}")
    print(f"CROSS-CLIENT {metric.upper()} MATRIX")
    print(f"{'='*60}")
    print("Rows = Source Models, Columns = Target Client Data")
    print("Diagonal = Own-client performance")
    print("-" * 60)
    
    # Header
    header = "Model\\Client"
    for j in range(num_clients):
        header += f"  Client{j:2d}"
    header += "    Avg"
    print(header)
    print("-" * 60)
    
    # Rows
    for i in range(num_clients):
        row = f"Model {i:2d}   "
        row_values = []
        for j in range(num_clients):
            value = matrix[i, j]
            if i == j:  # Own client - bold
                row += f"  {value:7.3f}"
            else:  # Cross client
                row += f"  {value:7.3f}"
            row_values.append(value)
        
        # Calculate average excluding own client
        cross_values = [matrix[i, j] for j in range(num_clients) if i != j]
        avg_cross = np.mean(cross_values) if cross_values else 0.0
        row += f"  {avg_cross:7.3f}"
        print(row)
    
    # Column averages
    print("-" * 60)
    avg_row = "Avg        "
    for j in range(num_clients):
        col_values = [matrix[i, j] for i in range(num_clients) if i != j]
        avg_col = np.mean(col_values) if col_values else 0.0
        avg_row += f"  {avg_col:7.3f}"
    
    # Overall cross-client average
    all_cross_values = []
    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                all_cross_values.append(matrix[i, j])
    
    overall_avg = np.mean(all_cross_values) if all_cross_values else 0.0
    avg_row += f"  {overall_avg:7.3f}"
    print(avg_row)
    
    print(f"{'='*60}")
    print(f"Overall Cross-Client Average {metric.capitalize()}: {overall_avg:.4f}")
    print(f"Own-Client Average {metric.capitalize()}: {np.mean(np.diag(matrix)):.4f}")
    print(f"{'='*60}")

# =============================================================================
# HYPERPARAMETER GRID SEARCH
# =============================================================================

def create_hyperparameter_grid():
    """Create hyperparameter grid for search"""

    hyperparameters = {
        'teacher_lr': [0.01, 0.001],
        'student_lr': [0.01, 0.001],
        'teacher_epochs': [150],
        'student_epochs': [80],
        'teacher_gamma': [0.95],
        'student_gamma': [0.95],
        'teacher_step': [15],
        'student_step': [12],
        'temperature': [3.0, 4.0, 5.0],
        'alpha': [0.7],  # KD loss weight
        'batch_size': [32, 64],
        'teacher_dropout': [0.2],
        'student_dropout': [0.2]
    }
    
    
    # Create all combinations
    keys = list(hyperparameters.keys())
    values = list(hyperparameters.values())
    
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations

def save_results_to_file(results_list, filename):
    """Save results to CSV file"""
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def create_streamlined_result_entry(params, exp_idx, num_clients, teacher_results, 
                                   client_results, avg_metrics):
    """
    Create streamlined result entry with best model statistics and cross-client matrix
    
    Args:
        params: Hyperparameters used
        exp_idx: Experiment index
        num_clients: Number of clients
        teacher_results: Teacher model results
        client_results: All client results
        avg_metrics: Aggregated metrics
    
    Returns:
        dict: Streamlined result entry for CSV
    """
    
    result_entry = {
        'experiment_id': exp_idx,
        'num_clients': num_clients,
        'timestamp': datetime.now().isoformat(),
        **params,  # All hyperparameters
        
        # Teacher performance
        'teacher_best_accuracy': teacher_results['best_accuracy'],
        'teacher_final_f1': teacher_results['final_f1'],
        'teacher_final_precision': teacher_results['final_precision'],
        'teacher_final_recall': teacher_results['final_recall'],
        
        # Summary metrics
        'avg_client_best_accuracy': avg_metrics['avg_best_accuracy'],
        'avg_client_final_f1': avg_metrics['avg_final_f1'],
        'avg_client_final_precision': avg_metrics['avg_final_precision'],
        'avg_client_final_recall': avg_metrics['avg_final_recall'],
        'avg_cross_client_accuracy': avg_metrics['avg_cross_client_accuracy'],
        'avg_cross_client_f1': avg_metrics['avg_cross_client_f1'],
        'avg_cross_client_precision': avg_metrics['avg_cross_client_precision'],
        'avg_cross_client_recall': avg_metrics['avg_cross_client_recall'],
        'std_cross_client_accuracy': avg_metrics['std_cross_client_accuracy'],
    }
    
    # Add individual client best performances (own data only)
    for i in range(num_clients):
        result_entry[f'client_{i}_best_accuracy'] = client_results[i]['best_accuracy']
        result_entry[f'client_{i}_final_f1'] = client_results[i]['final_f1']
        result_entry[f'client_{i}_final_precision'] = client_results[i]['final_precision']
        result_entry[f'client_{i}_final_recall'] = client_results[i]['final_recall']
    
    # Add cross-client matrix values (accuracy only for space efficiency)
    # Format: matrix_i_j means student model i evaluated on client j validation data
    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                # Own client performance (diagonal)
                result_entry[f'matrix_{i}_{j}'] = client_results[i]['best_accuracy']
            else:
                # Cross-client performance
                individual_results = client_results[i]['cross_client_evaluation']['individual_results']
                cross_key = f'cross_client_{j}'
                if cross_key in individual_results:
                    result_entry[f'matrix_{i}_{j}'] = individual_results[cross_key]['accuracy']
                else:
                    result_entry[f'matrix_{i}_{j}'] = np.nan
    
    return result_entry

def run_single_experiment(params, user_path='femnist_dataset', num_clients=7, seed=42, 
                         device=None, results_dir="results"):
    """Run a single experiment with given hyperparameters including cross-client evaluation"""
    
    fix_all_seeds(seed)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running experiment with parameters: {params}")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load validation data with split for teacher
    teacher_train_loader, teacher_val_loader = load_validation_data_with_split(
        user_path, params['batch_size']
    )

    # Load all client data for cross-evaluation
    all_client_data = load_all_client_data(user_path, params['batch_size'], num_clients)

    print("\n=== Teacher Data ===")
    print(f"Teacher train samples: {len(teacher_train_loader.dataset)}")
    print(f"Teacher val samples:   {len(teacher_val_loader.dataset)}")
    
    # Initialize and train teacher
    teacher_model = TeacherNetwork(
        input_size=1280,
        hidden_sizes=[512, 256, 128],
        num_classes=4,
        dropout_rate=params['teacher_dropout'],
        device=device
    )
    
    teacher_model, teacher_results = train_teacher_with_best_model_selection(
        teacher_model, 
        teacher_train_loader,
        teacher_val_loader,
        device, 
        epochs=params['teacher_epochs'],
        lr=params['teacher_lr'],
        gamma=params['teacher_gamma'],
        step=params['teacher_step']
    )
    
    # Train students for each client
    all_client_results = []
    all_cross_client_results = []
    
    for client_id in range(num_clients):
        print(f"\nTraining client {client_id}...")
        
        # Get client data
        train_loader = all_client_data[client_id]['train_loader']
        val_loader = all_client_data[client_id]['val_loader']

        print("=== Student Data ===")
        print(f"Student train samples: {len(train_loader.dataset)}")
        print(f"Student val samples:   {len(val_loader.dataset)}")
        
        # Initialize student
        student_model = StudentNetwork(
            input_size=1280,
            hidden_sizes=[256, 128],
            num_classes=4,
            dropout_rate=params['student_dropout'],
            device=device
        )
        
        # Train student with KD
        student_model, client_results = train_student_with_best_model_selection(
            student_model, 
            teacher_model, 
            train_loader, 
            val_loader,
            device, 
            epochs=params['student_epochs'],
            lr=params['student_lr'],
            temperature=params['temperature'],
            alpha=params['alpha'],
            gamma=params['student_gamma'],
            step=params['student_step'],
            client_id=client_id
        )
        
        # Evaluate student on other clients' data
        cross_client_results = evaluate_student_on_other_clients(
            student_model, client_id, all_client_data, device, num_clients
        )
        
        # Combine results
        client_results['cross_client_evaluation'] = cross_client_results
        
        all_client_results.append(client_results)
        all_cross_client_results.append(cross_client_results)
    
    # Calculate average metrics across all clients (using best models)
    client_best_accuracies = [cr['best_accuracy'] for cr in all_client_results]
    client_final_f1s = [cr['final_f1'] for cr in all_client_results]
    client_final_precisions = [cr['final_precision'] for cr in all_client_results]
    client_final_recalls = [cr['final_recall'] for cr in all_client_results]
    client_final_losses = [cr['final_loss'] for cr in all_client_results]
    
    # Calculate cross-client metrics
    cross_client_accuracies = [ccr['avg_cross_accuracy'] for ccr in all_cross_client_results]
    cross_client_f1s = [ccr['avg_cross_f1'] for ccr in all_cross_client_results]
    cross_client_precisions = [ccr['avg_cross_precision'] for ccr in all_cross_client_results]
    cross_client_recalls = [ccr['avg_cross_recall'] for ccr in all_cross_client_results]
    cross_client_losses = [ccr['avg_cross_loss'] for ccr in all_cross_client_results]
    
    avg_metrics = {
        # Own client performance
        'avg_best_accuracy': np.mean(client_best_accuracies),
        'avg_final_f1': np.mean(client_final_f1s),
        'avg_final_precision': np.mean(client_final_precisions),
        'avg_final_recall': np.mean(client_final_recalls),
        'avg_final_loss': np.mean(client_final_losses),
        'std_best_accuracy': np.std(client_best_accuracies),
        
        # Cross-client performance
        'avg_cross_client_accuracy': np.mean(cross_client_accuracies),
        'avg_cross_client_f1': np.mean(cross_client_f1s),
        'avg_cross_client_precision': np.mean(cross_client_precisions),
        'avg_cross_client_recall': np.mean(cross_client_recalls),
        'avg_cross_client_loss': np.mean(cross_client_losses),
        'std_cross_client_accuracy': np.std(cross_client_accuracies),
        
        # Average of averages (mean cross-client performance across all students)
        'mean_of_cross_client_averages': np.mean(cross_client_accuracies)
    }
    
    print(f"\n=== EXPERIMENT SUMMARY ===")
    print(f"Average own-client accuracy: {avg_metrics['avg_best_accuracy']:.4f}")
    print(f"Average cross-client accuracy: {avg_metrics['avg_cross_client_accuracy']:.4f}")
    print(f"Mean of cross-client averages: {avg_metrics['mean_of_cross_client_averages']:.4f}")
    
    # Create and display cross-client matrices (print only, no plots)
    print(f"\n=== CROSS-CLIENT PERFORMANCE MATRICES ===")
    
    # Create matrices for different metrics
    accuracy_matrix = create_cross_client_matrix(all_client_results, num_clients, 'accuracy')
    f1_matrix = create_cross_client_matrix(all_client_results, num_clients, 'f1')
    
    # Print summary tables
    print_cross_client_summary_table(accuracy_matrix, num_clients, 'accuracy')
    print_cross_client_summary_table(f1_matrix, num_clients, 'f1')
    
    return teacher_results, all_client_results, avg_metrics

def run_grid_search(user_path='femnist_dataset', num_clients=7, seed=42, max_experiments=20):
    """Run grid search over hyperparameters with continuous saving (no plotting)"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Grid Search with {num_clients} clients on {device}")
    print("=" * 80)
    
    # Create results directory
    results_dir = f"irds_results_clients"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get hyperparameter combinations
    param_combinations = create_hyperparameter_grid()
    print(f"Total combinations: {len(param_combinations)}")
    
    # Limit experiments if needed
    if max_experiments > 0 and len(param_combinations) > max_experiments:
        param_combinations = random.sample(param_combinations, max_experiments)
        print(f"Limited to {max_experiments} random combinations")
    
    # Results storage
    all_results = []
    
    # Create results filename
    results_filename = f'{results_dir}/kd_cross_client_results_{num_clients}_clients.csv'
    
    best_cross_accuracy = 0.0
    best_experiment_idx = None
    
    for exp_idx, params in enumerate(param_combinations):
        print(f"\nExperiment {exp_idx+1}/{len(param_combinations)}")
        print("-" * 50)
        print("Hyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        try:
            teacher_results, client_results, avg_metrics = run_single_experiment(
                params, user_path, num_clients, seed + exp_idx, device, results_dir
            )
            
            # Create streamlined result entry
            result_entry = create_streamlined_result_entry(
                params, exp_idx, num_clients, teacher_results, client_results, avg_metrics
            )
            
            all_results.append(result_entry)
            
            # Check if this is the best cross-client performance
            current_cross_accuracy = avg_metrics['avg_cross_client_accuracy']
            if current_cross_accuracy > best_cross_accuracy:
                best_cross_accuracy = current_cross_accuracy
                best_experiment_idx = exp_idx
            
            # Save results after each experiment
            save_results_to_file(all_results, results_filename)
            
            print(f"Results Summary:")
            print(f"  Teacher Best Acc: {teacher_results['best_accuracy']:.4f}")
            print(f"  Avg Client Best Acc (own data): {avg_metrics['avg_best_accuracy']:.4f}")
            print(f"  Avg Cross-Client Acc: {avg_metrics['avg_cross_client_accuracy']:.4f}")
            print(f"  Mean of Cross-Client Averages: {avg_metrics['mean_of_cross_client_averages']:.4f}")
            
        except Exception as e:
            print(f"Experiment {exp_idx+1} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print best results
    if len(all_results) > 0:
        best_result_own = max(all_results, key=lambda x: x['avg_client_best_accuracy'])
        best_result_cross = max(all_results, key=lambda x: x['avg_cross_client_accuracy'])
        
        print("\n" + "=" * 80)
        print("BEST CONFIGURATIONS FOUND:")
        print("=" * 80)
        
        print(f"\nBest for Own-Client Performance (Experiment {best_result_own['experiment_id']}):")
        print(f"  Average Client Best Accuracy: {best_result_own['avg_client_best_accuracy']:.4f}")
        print(f"  Average Cross-Client Accuracy: {best_result_own['avg_cross_client_accuracy']:.4f}")
        print(f"  Teacher Best Accuracy: {best_result_own['teacher_best_accuracy']:.4f}")
        
        print(f"\nBest for Cross-Client Performance (Experiment {best_result_cross['experiment_id']}):")
        print(f"  Average Cross-Client Accuracy: {best_result_cross['avg_cross_client_accuracy']:.4f}")
        print(f"  Average Client Best Accuracy: {best_result_cross['avg_client_best_accuracy']:.4f}")
        print(f"  Teacher Best Accuracy: {best_result_cross['teacher_best_accuracy']:.4f}")
        
        print("\nBest Hyperparameters (Cross-Client):")
        hyperparams = ['teacher_lr', 'student_lr', 'teacher_epochs', 'student_epochs',
                      'teacher_gamma', 'student_gamma', 'teacher_step', 'student_step',
                      'temperature', 'alpha', 'batch_size', 'teacher_dropout', 'student_dropout']
        for param in hyperparams:
            if param in best_result_cross:
                print(f"  {param}: {best_result_cross[param]}")
        
        # Print cross-client matrix for best experiment
        print(f"\nCross-Client Matrix for Best Experiment (ID: {best_result_cross['experiment_id']}):")
        print("Matrix format: matrix_i_j = Student model i evaluated on Client j validation data")
        print("-" * 60)
        
        # Print matrix header
        header = "Model\\Client"
        for j in range(num_clients):
            header += f"  Client{j:2d}"
        print(header)
        print("-" * 60)
        
        # Print matrix rows
        for i in range(num_clients):
            row = f"Model {i:2d}   "
            for j in range(num_clients):
                value = best_result_cross[f'matrix_{i}_{j}']
                row += f"  {value:7.3f}"
            print(row)
        
        print(f"\nFinal results saved to: {results_filename}")
    
    return all_results, results_filename, results_dir

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_kd_experiments(user_path='femnist_dataset', client_configs=[3, 5, 7], seed=42, 
                      max_experiments_per_config=10):
    """
    Run knowledge distillation experiments for different client configurations
    with comprehensive cross-client analysis (no plotting)
    """
    
    all_experiments = []
    
    for num_clients in client_configs:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENTS WITH {num_clients} CLIENTS")
        print(f"{'='*80}")
        
        results, filename, results_dir = run_grid_search(
            user_path=user_path,
            num_clients=num_clients,
            seed=seed,
            max_experiments=max_experiments_per_config
        )
        
        all_experiments.extend(results)
        print(f"Completed experiments for {num_clients} clients.")
        print(f"Results saved to {filename}")

    return all_experiments

# =============================================================================
# UTILITY FUNCTIONS FOR ANALYSIS
# =============================================================================

def analyze_cross_client_results(csv_filename, num_clients):
    """
    Analyze cross-client results from saved CSV file
    
    Args:
        csv_filename: Path to the results CSV file
        num_clients: Number of clients used in experiments
    """
    
    print(f"Analyzing results from {csv_filename}")
    print("=" * 60)
    
    # Load results
    df = pd.read_csv(csv_filename)
    
    if df.empty:
        print("No results found in CSV file!")
        return
    
    # Find best experiment for cross-client performance
    best_idx = df['avg_cross_client_accuracy'].idxmax()
    best_exp = df.loc[best_idx]
    
    print(f"Best Experiment (ID: {best_exp['experiment_id']}):")
    print(f"  Cross-Client Accuracy: {best_exp['avg_cross_client_accuracy']:.4f}")
    print(f"  Own-Client Accuracy: {best_exp['avg_client_best_accuracy']:.4f}")
    print(f"  Teacher Accuracy: {best_exp['teacher_best_accuracy']:.4f}")
    
    # Reconstruct cross-client matrix for best experiment
    matrix = np.zeros((num_clients, num_clients))
    
    for i in range(num_clients):
        for j in range(num_clients):
            col_name = f'matrix_{i}_{j}'
            if col_name in df.columns:
                matrix[i, j] = best_exp[col_name]
            else:
                print(f"Warning: Column {col_name} not found")
    
    print(f"\nCross-Client Matrix for Best Experiment:")
    print_cross_client_summary_table(matrix, num_clients, 'accuracy')
    
    # Summary statistics
    print(f"\nSummary Statistics across all experiments:")
    print(f"  Mean Cross-Client Accuracy: {df['avg_cross_client_accuracy'].mean():.4f}")
    print(f"  Std Cross-Client Accuracy: {df['avg_cross_client_accuracy'].std():.4f}")
    print(f"  Max Cross-Client Accuracy: {df['avg_cross_client_accuracy'].max():.4f}")
    print(f"  Min Cross-Client Accuracy: {df['avg_cross_client_accuracy'].min():.4f}")
    
    return df, matrix

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Run experiments for different client configurations (no plotting)
    all_results = run_kd_experiments(
        user_path='/scratch.hpc/mazza/femnist/irds_lab/irds_dataset/',
        client_configs=[5],  # Start with 3 clients for testing
        seed=42,
        max_experiments_per_config=100  # Reduced for testing
    )
    
    print("\nAll experiments completed!")
    
    # Example of how to analyze results afterwards
    # analyze_cross_client_results('results_3_clients/kd_cross_client_results_3_clients.csv', 3)