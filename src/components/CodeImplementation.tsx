import React, { useState } from 'react';
import { Code2, FileText, Download, Play } from 'lucide-react';

const CodeImplementation: React.FC = () => {
  const [activeFile, setActiveFile] = useState('teacher');

  const codeFiles = {
    teacher: {
      title: 'teacher_model.py',
      language: 'python',
      content: `import torch
import torch.nn as nn
import torchvision.models as models

class TeacherModel(nn.Module):
    """
    ResNet-18 based teacher model for CIFAR-10
    High capacity model with 11.2M parameters
    """
    def __init__(self, num_classes=10):
        super(TeacherModel, self).__init__()
        # Load pre-trained ResNet-18 and modify for CIFAR-10
        self.model = models.resnet18(pretrained=False)
        
        # Modify the first conv layer for 32x32 input
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, 
                                   stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool for small images
        
        # Modify final layer for CIFAR-10 (10 classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def create_teacher_model(device='cuda'):
    """Create and initialize teacher model"""
    model = TeacherModel(num_classes=10)
    model = model.to(device)
    return model

# Training configuration
def get_teacher_config():
    return {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'epochs': 200,
        'batch_size': 128,
        'scheduler': 'cosine'
    }`
    },
    student: {
      title: 'student_model.py',
      language: 'python',
      content: `import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentModel(nn.Module):
    """
    Lightweight CNN student model for CIFAR-10
    Only 0.2M parameters - 56x smaller than teacher
    """
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 32x32 -> 16x16
        
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 16x16 -> 8x8
        
        # Global average pooling
        x = self.global_avg_pool(x)  # 8x8 -> 1x1
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    
def create_student_model(device='cuda'):
    """Create and initialize student model"""
    model = StudentModel(num_classes=10)
    model = model.to(device)
    return model

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)`
    },
    distillation: {
      title: 'knowledge_distillation.py',
      language: 'python',
      content: `import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining:
    - KL Divergence loss (soft targets from teacher)
    - Cross Entropy loss (hard labels from dataset)
    """
    def __init__(self, alpha=0.7, temperature=4.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets (knowledge distillation loss)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        distillation_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Hard targets (standard classification loss)
        classification_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = (self.alpha * distillation_loss + 
                     (1 - self.alpha) * classification_loss)
        
        return total_loss, distillation_loss, classification_loss

def distill_knowledge(teacher_model, student_model, train_loader, 
                     val_loader, device='cuda', epochs=100):
    """
    Main knowledge distillation training loop
    """
    # Set teacher to evaluation mode (frozen)
    teacher_model.eval()
    
    # Initialize student optimizer and loss
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = DistillationLoss(alpha=0.7, temperature=4.0)
    
    # Training history
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        student_model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass through both models
            with torch.no_grad():
                teacher_logits = teacher_model(data)
            
            student_logits = student_model(data)
            
            # Calculate distillation loss
            loss, kd_loss, ce_loss = criterion(student_logits, teacher_logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        val_acc = evaluate_model(student_model, val_loader, device)
        
        train_losses.append(epoch_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss={epoch_loss:.4f}, Val Acc={val_acc:.4f}')
    
    return train_losses, val_accuracies

def evaluate_model(model, data_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total`
    },
    training: {
      title: 'train.py',
      language: 'python',
      content: `import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

from teacher_model import create_teacher_model, get_teacher_config
from student_model import create_student_model, count_parameters
from knowledge_distillation import distill_knowledge, evaluate_model

def load_cifar10_data():
    """Load and prepare CIFAR-10 dataset"""
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # No augmentation for validation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_test)
    
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def train_teacher(teacher_model, trainloader, testloader, device='cuda'):
    """Train the teacher model"""
    print("Training Teacher Model (ResNet-18)...")
    
    config = get_teacher_config()
    optimizer = optim.SGD(teacher_model.parameters(), 
                         lr=config['lr'], 
                         momentum=config['momentum'],
                         weight_decay=config['weight_decay'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    criterion = torch.nn.CrossEntropyLoss()
    
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(config['epochs']):
        teacher_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(trainloader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = teacher_model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = correct / total
        val_acc = evaluate_model(teacher_model, testloader, device)
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}')
    
    return train_accuracies, val_accuracies

def main():
    """Main training pipeline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    trainloader, testloader = load_cifar10_data()
    
    # Create models
    teacher_model = create_teacher_model(device)
    student_model = create_student_model(device)
    
    print(f"Teacher parameters: {count_parameters(teacher_model):,}")
    print(f"Student parameters: {count_parameters(student_model):,}")
    
    # Train teacher model
    teacher_train_acc, teacher_val_acc = train_teacher(
        teacher_model, trainloader, testloader, device)
    
    print(f"Teacher final accuracy: {teacher_val_acc[-1]:.4f}")
    
    # Save teacher model
    torch.save(teacher_model.state_dict(), 'teacher_model.pth')
    
    # Knowledge distillation
    print("\\nStarting Knowledge Distillation...")
    student_losses, student_accuracies = distill_knowledge(
        teacher_model, student_model, trainloader, testloader, device)
    
    print(f"Student final accuracy: {student_accuracies[-1]:.4f}")
    
    # Save student model
    torch.save(student_model.state_dict(), 'student_model.pth')
    
    # Performance comparison
    teacher_final_acc = teacher_val_acc[-1]
    student_final_acc = student_accuracies[-1]
    
    print("\\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Teacher Accuracy: {teacher_final_acc:.4f}")
    print(f"Student Accuracy: {student_final_acc:.4f}")
    print(f"Accuracy Drop: {teacher_final_acc - student_final_acc:.4f}")
    print(f"Parameter Reduction: {count_parameters(teacher_model) / count_parameters(student_model):.1f}x")
    print("="*50)

if __name__ == "__main__":
    main()`
    },
    utils: {
      title: 'utils.py',
      language: 'python',
      content: `import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def measure_inference_time(model, input_size=(1, 3, 32, 32), device='cuda', num_runs=100):
    """Measure average inference time of a model"""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Measure time
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
    return avg_time

def plot_training_curves(teacher_acc, student_acc, teacher_loss=None, student_loss=None):
    """Plot training curves comparison"""
    fig, axes = plt.subplots(1, 2 if teacher_loss is None else 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(teacher_acc, label='Teacher', color='blue', linewidth=2)
    axes[0].plot(student_acc, label='Student', color='orange', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot (if available)
    if teacher_loss is not None and student_loss is not None:
        axes[1].plot(teacher_loss, label='Teacher', color='blue', linewidth=2)
        axes[1].plot(student_loss, label='Student', color='orange', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comparison_table(teacher_model, student_model, teacher_acc, student_acc, device='cuda'):
    """Create detailed comparison table"""
    
    # Count parameters
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    # Measure inference time
    teacher_time = measure_inference_time(teacher_model, device=device)
    student_time = measure_inference_time(student_model, device=device)
    
    # Calculate model sizes (approximate)
    teacher_size = teacher_params * 4 / (1024**2)  # MB (assuming float32)
    student_size = student_params * 4 / (1024**2)  # MB
    
    # Create comparison data
    comparison_data = {
        'Metric': [
            'Parameters', 'Model Size (MB)', 'Accuracy (%)', 
            'Inference Time (ms)', 'Speed Improvement', 'Parameter Reduction'
        ],
        'Teacher (ResNet-18)': [
            f'{teacher_params:,}', f'{teacher_size:.1f}', f'{teacher_acc*100:.1f}',
            f'{teacher_time:.2f}', '1.0x', '1.0x'
        ],
        'Student (Custom CNN)': [
            f'{student_params:,}', f'{student_size:.1f}', f'{student_acc*100:.1f}',
            f'{student_time:.2f}', f'{teacher_time/student_time:.1f}x', 
            f'{teacher_params/student_params:.1f}x'
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    return df

def visualize_model_comparison(comparison_df):
    """Create visualization of model comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract numeric values for plotting
    teacher_params = float(comparison_df.iloc[0, 1].replace(',', ''))
    student_params = float(comparison_df.iloc[0, 2].replace(',', ''))
    
    teacher_size = float(comparison_df.iloc[1, 1])
    student_size = float(comparison_df.iloc[1, 2])
    
    teacher_acc = float(comparison_df.iloc[2, 1])
    student_acc = float(comparison_df.iloc[2, 2])
    
    teacher_time = float(comparison_df.iloc[3, 1])
    student_time = float(comparison_df.iloc[3, 2])
    
    # Parameters comparison
    axes[0, 0].bar(['Teacher', 'Student'], [teacher_params, student_params], 
                   color=['#3B82F6', '#F59E0B'])
    axes[0, 0].set_ylabel('Parameters')
    axes[0, 0].set_title('Model Parameters')
    axes[0, 0].set_yscale('log')
    
    # Model size comparison
    axes[0, 1].bar(['Teacher', 'Student'], [teacher_size, student_size], 
                   color=['#3B82F6', '#F59E0B'])
    axes[0, 1].set_ylabel('Size (MB)')
    axes[0, 1].set_title('Model Size')
    
    # Accuracy comparison
    axes[1, 0].bar(['Teacher', 'Student'], [teacher_acc, student_acc], 
                   color=['#3B82F6', '#F59E0B'])
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Model Accuracy')
    axes[1, 0].set_ylim(80, 95)
    
    # Inference time comparison
    axes[1, 1].bar(['Teacher', 'Student'], [teacher_time, student_time], 
                   color=['#3B82F6', '#F59E0B'])
    axes[1, 1].set_ylabel('Time (ms)')
    axes[1, 1].set_title('Inference Time')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(teacher_model, student_model, teacher_acc, student_acc, 
                train_history, output_dir='results/'):
    """Save all results and models"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save models
    torch.save(teacher_model.state_dict(), f'{output_dir}/teacher_model.pth')
    torch.save(student_model.state_dict(), f'{output_dir}/student_model.pth')
    
    # Save training history
    np.save(f'{output_dir}/training_history.npy', train_history)
    
    # Create and save comparison table
    comparison_df = create_comparison_table(teacher_model, student_model, 
                                          teacher_acc, student_acc)
    comparison_df.to_csv(f'{output_dir}/comparison_results.csv', index=False)
    
    print(f"Results saved to {output_dir}")
    return comparison_df`
    }
  };

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-green-900/50 to-blue-900/50 rounded-xl p-8 border border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Code2 className="w-8 h-8 text-green-400" />
            <div>
              <h2 className="text-3xl font-bold">Complete Implementation</h2>
              <p className="text-gray-300">Production-ready PyTorch code for knowledge distillation</p>
            </div>
          </div>
          <div className="flex space-x-3">
            <button className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition-colors">
              <Download className="w-4 h-4" />
              <span>Download All</span>
            </button>
            <button className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition-colors">
              <Play className="w-4 h-4" />
              <span>Run Code</span>
            </button>
          </div>
        </div>
      </div>

      {/* File Navigation */}
      <div className="bg-gray-800/50 rounded-lg border border-gray-700 overflow-hidden">
        <div className="flex border-b border-gray-700">
          {Object.entries(codeFiles).map(([key, file]) => (
            <button
              key={key}
              onClick={() => setActiveFile(key)}
              className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeFile === key
                  ? 'border-blue-500 text-blue-400 bg-gray-900/50'
                  : 'border-transparent text-gray-400 hover:text-gray-300 hover:bg-gray-900/25'
              }`}
            >
              <FileText className="w-4 h-4" />
              <span>{file.title}</span>
            </button>
          ))}
        </div>

        <div className="p-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">{codeFiles[activeFile].title}</h3>
            <button className="flex items-center space-x-2 bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded text-sm transition-colors">
              <Download className="w-3 h-3" />
              <span>Download</span>
            </button>
          </div>
          
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="language-python p-4 text-sm overflow-x-auto">
              <code className="language-python text-gray-300 leading-relaxed">
                {codeFiles[activeFile].content}
              </code>
            </pre>
          </div>
        </div>
      </div>

      {/* Implementation Steps */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-xl font-semibold mb-6">Implementation Steps</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="bg-gray-900/50 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="bg-blue-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold">1</div>
                <h4 className="font-semibold">Setup Environment</h4>
              </div>
              <p className="text-sm text-gray-300">Install PyTorch, torchvision, and other dependencies</p>
            </div>
            
            <div className="bg-gray-900/50 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="bg-blue-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold">2</div>
                <h4 className="font-semibold">Train Teacher Model</h4>
              </div>
              <p className="text-sm text-gray-300">Train ResNet-18 on CIFAR-10 to achieve high accuracy</p>
            </div>
            
            <div className="bg-gray-900/50 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="bg-blue-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold">3</div>
                <h4 className="font-semibold">Design Student Model</h4>
              </div>
              <p className="text-sm text-gray-300">Create lightweight CNN with significantly fewer parameters</p>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="bg-gray-900/50 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="bg-orange-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold">4</div>
                <h4 className="font-semibold">Knowledge Distillation</h4>
              </div>
              <p className="text-sm text-gray-300">Train student using soft targets from teacher model</p>
            </div>
            
            <div className="bg-gray-900/50 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="bg-orange-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold">5</div>
                <h4 className="font-semibold">Evaluation</h4>
              </div>
              <p className="text-sm text-gray-300">Compare performance, speed, and model size metrics</p>
            </div>
            
            <div className="bg-gray-900/50 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="bg-orange-600 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold">6</div>
                <h4 className="font-semibold">Analysis & Export</h4>
              </div>
              <p className="text-sm text-gray-300">Generate visualizations and export results</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CodeImplementation;