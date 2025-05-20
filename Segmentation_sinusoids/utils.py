import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
#from torchvision.utils import make_grid
from preprocess.dataset_augumented import LiverVolumeDataset  
import random
from tqdm import tqdm
from config import LOSS_FUNCTION

def dice_coefficient(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    intersection = (output * target).sum()
    union = output.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def iou_score(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    intersection = (output * target).sum()
    union = output.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def precision_score(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    true_positives = (output * target).sum()
    predicted_positives = output.sum()
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    return precision.item()

def recall_score(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    true_positives = (output * target).sum()
    actual_positives = target.sum()
    recall = (true_positives + smooth) / (actual_positives + smooth)
    return recall.item()

def f1_score(output, target):
    precision = precision_score(output, target)
    recall = recall_score(output, target)
    f1 = (2 * precision * recall) / (precision + recall + 1e-5)  # Add small constant to avoid division by zero
    return f1


# Custom weight initialization
def initialize_weights(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def initialize_tensorboard(LOG_DIR, model, train_loader, DEVICE):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    writer = SummaryWriter(log_dir=LOG_DIR)
    sample_input, _ = next(iter(train_loader))  # Get a sample input
    sample_input = sample_input.to(DEVICE)
    writer.add_graph(model, sample_input)
    return writer

def log_validation_metrics(writer, epoch, avg_val_loss, avg_val_dice, avg_val_iou,
                           avg_val_precision, avg_val_recall, avg_val_f1):
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    writer.add_scalar("Dice/Validation", avg_val_dice, epoch)
    writer.add_scalar("IoU/Validation", avg_val_iou, epoch)
    writer.add_scalar("Precision/Validation", avg_val_precision, epoch)
    writer.add_scalar("Recall/Validation", avg_val_recall, epoch)
    writer.add_scalar("F1/Validation", avg_val_f1, epoch)

def log_model_parameters(writer, model, epoch):
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
        if param.grad is not None:
            writer.add_histogram(f"{name}.grad", param.grad, epoch)

def log_learning_rate(writer, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        writer.add_scalar(f"Learning Rate/Group_{i}", param_group['lr'], epoch)

def close_tensorboard(writer):
    writer.close()

def calculate_and_accumulate_metrics(outputs, labels, metrics_dict):
    dice = dice_coefficient(outputs, labels)
    iou = iou_score(outputs, labels)
    precision = precision_score(outputs, labels)
    recall = recall_score(outputs, labels)
    f1 = f1_score(outputs, labels)

    metrics_dict['dice'] += dice
    metrics_dict['iou'] += iou
    metrics_dict['precision'] += precision
    metrics_dict['recall'] += recall
    metrics_dict['f1'] += f1

    return metrics_dict

def update_validation_progress_bar(pbar, i, val_loss, metrics_dict):
    pbar.set_postfix(val_loss=val_loss / (i + 1),
                     dice=metrics_dict['dice'] / (i + 1),
                     iou=metrics_dict['iou'] / (i + 1),
                     precision=metrics_dict['precision'] / (i + 1),
                     recall=metrics_dict['recall'] / (i + 1),
                     f1=metrics_dict['f1'] / (i + 1))
    pbar.update(1)


def compute_avg_validation_metrics(val_loss, metrics_dict, val_loader_length):
    avg_val_loss = val_loss / val_loader_length
    avg_metrics = {key: value / val_loader_length for key, value in metrics_dict.items()}
    return avg_val_loss, avg_metrics

def print_metrics_val(epoch, NUM_EPOCHS, avg_val_loss, avg_metrics):
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}, "
          f"Dice: {avg_metrics['dice']:.4f}, IoU: {avg_metrics['iou']:.4f}, "
          f"Precision: {avg_metrics['precision']:.4f}, Recall: {avg_metrics['recall']:.4f}, "
          f"F1: {avg_metrics['f1']:.4f}", flush=True)

def calculate_print_log_train_metrics(writer, epoch, NUM_EPOCHS, train_loader, running_loss, running_dice, running_iou,running_precision, running_recall, running_f1):
    avg_training_loss = running_loss / len(train_loader)
    avg_training_dice = running_dice / len(train_loader)
    avg_training_iou = running_iou / len(train_loader)
    avg_training_precision = running_precision / len(train_loader)
    avg_training_recall = running_recall / len(train_loader)
    avg_training_f1 = running_f1 / len(train_loader)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {avg_training_loss:.4f}, "
          f"Dice: {avg_training_dice:.4f}, IoU: {avg_training_iou:.4f}, "
          f"Precision: {avg_training_precision:.4f}, Recall: {avg_training_recall:.4f}, "
          f"F1: {avg_training_f1:.4f}", flush=True)
    
    writer.add_scalar("Loss/Train", avg_training_loss, epoch)
    writer.add_scalar("Dice/Train", avg_training_dice, epoch)
    writer.add_scalar("IoU/Train", avg_training_iou, epoch)
    writer.add_scalar("Precision/Train", avg_training_precision, epoch)
    writer.add_scalar("Recall/Train", avg_training_recall, epoch)
    writer.add_scalar("F1/Train", avg_training_f1, epoch)
    return


def visualize_predictions(inputs, labels, outputs, writer, epoch):
        outputs_sigmoid = torch.sigmoid(outputs)
        
        random_batch = random.randint(0, inputs.shape[0] - 1)  # Select a random batch
        random_volume = random.randint(0, inputs.shape[1] - 1)  # Select a random volume from the batch
        slice_idx = random.randint(0, inputs.shape[2] - 1)  # Select the middle slice
        # Get the data from tensors
        input_slice = inputs[random_batch, random_volume, slice_idx, :, :].cpu().numpy()
        label_slice = labels[random_batch, random_volume, slice_idx, :, :].cpu().numpy()
        output_slice = outputs_sigmoid[random_batch, random_volume, slice_idx, :, :].cpu().numpy()
        # Normalize images for visualization
        input_slice_norm = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min() + 1e-5)
        output_slice_norm = (output_slice - output_slice.min()) / (output_slice.max() - output_slice.min() + 1e-5)

        # Add images to TensorBoard
        writer.add_images('Input image', input_slice_norm, epoch, dataformats='HW')
        writer.add_images('Label image', label_slice, epoch, dataformats='HW')
        writer.add_images('Predicted image', output_slice_norm, epoch, dataformats='HW')

def evaluate_test_set(model, test_loader, device, writer=None, epoch=None):
    model.eval()
    test_loss = 0.0
    metrics_dict = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Test Evaluation", unit="batch") as pbar:
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.float()
                outputs = model(inputs)
                if labels.shape != outputs.shape:
                    labels = F.interpolate(labels, size=outputs.shape[2:], mode='trilinear')
                loss = LOSS_FUNCTION(outputs, labels)
                test_loss += loss.item()
                metrics_dict = calculate_and_accumulate_metrics(outputs, labels, metrics_dict)
                
                # Visualize first batch of test predictions
                if i == 0 and writer is not None:
                    visualize_predictions(inputs, labels, outputs, writer, epoch)
                
                pbar.set_postfix(
                    loss=test_loss / (i + 1),
                    dice=metrics_dict['dice'] / (i + 1),
                    iou=metrics_dict['iou'] / (i + 1)
                )
                pbar.update(1)
    
    avg_test_loss = test_loss / len(test_loader)
    avg_metrics = {k: v / len(test_loader) for k, v in metrics_dict.items()}
    
    # Print metrics
    print("\nFinal Test Metrics:")
    print(f"Loss: {avg_test_loss:.4f}")
    print(f"Dice: {avg_metrics['dice']:.4f}")
    print(f"IoU: {avg_metrics['iou']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"F1: {avg_metrics['f1']:.4f}")
    
    # Log to TensorBoard if writer is provided
    if writer is not None:
        writer.add_scalar('Test/Loss', avg_test_loss, 
                        global_step=epoch if epoch is not None else 0)
        writer.add_scalar('Test/Dice', avg_metrics['dice'], 
                        global_step=epoch if epoch is not None else 0)
        writer.add_scalar('Test/IoU', avg_metrics['iou'], 
                        global_step=epoch if epoch is not None else 0)
        writer.add_scalar('Test/Precision', avg_metrics['precision'], 
                        global_step=epoch if epoch is not None else 0)
        writer.add_scalar('Test/Recall', avg_metrics['recall'], 
                        global_step=epoch if epoch is not None else 0)
        writer.add_scalar('Test/F1', avg_metrics['f1'], 
                        global_step=epoch if epoch is not None else 0)
        writer.flush()
    
    return avg_test_loss, avg_metrics