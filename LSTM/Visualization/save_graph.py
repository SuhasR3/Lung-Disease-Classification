import matplotlib.pyplot as plt
import re

# Function to extract data from text file
def extract_data(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    epochs = []
    training_losses = []
    validation_losses = []
    f1_scores = []

    # Use regex to extract values from each line
    for line in data:
        epoch_match = re.search(r'Epoch (\d+)', line)
        train_loss_match = re.search(r'Average Training Loss: ([\d.]+)', line)
        val_loss_match = re.search(r'Average Validation Loss: ([\d.]+)', line)
        f1_score_match = re.search(r'Validation F1-Score: ([\d.]+)', line)

        if epoch_match and train_loss_match and val_loss_match and f1_score_match:
            epoch = int(epoch_match.group(1))
            training_loss = float(train_loss_match.group(1))
            validation_loss = float(val_loss_match.group(1))
            f1_score = float(f1_score_match.group(1))
            
            epochs.append(epoch)
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)
            f1_scores.append(f1_score)
    
    return epochs, training_losses, validation_losses, f1_scores

# List of files
files = {
    'Adadelta Baseline':    './baseline/lstm_baseline_adadelta.txt',
    'AdamW Baseline':       './baseline/lstm_baseline_adamw.txt',
    'SGD Baseline':         './baseline/lstm_baseline_sgd.txt',
    'Adadelta HP1':         './hyperparameters1/lstm_hp1_adadelta.txt',
    'AdamW HP1':            './hyperparameters1/lstm_hp1_adamw.txt',
    'SGD HP1':              './hyperparameters1/lstm_hp1_sgd.txt',
    'Adadelta HP2':         './hyperparameters2/lstm_hp2_adadelta.txt',
    'AdamW HP2':            './hyperparameters2/lstm_hp2_adamw.txt',
    'SGD HP2':              './hyperparameters2/lstm_hp2_sgd.txt',
    'Adadelta S':           './scheduler/lstm_scheduler_adadelta.txt',
    'AdamW S':              './scheduler/lstm_scheduler_adamw.txt',
    'SGD S':                './scheduler/lstm_scheduler_sgd.txt'
}

baseline_files = {
    'Adadelta Baseline':    './baseline/lstm_baseline_adadelta.txt',
    'AdamW Baseline':       './baseline/lstm_baseline_adamw.txt',
    'SGD Baseline':         './baseline/lstm_baseline_sgd.txt'
}

hp1_files = {
    'Adadelta HP1': './hyperparameters1/lstm_hp1_adadelta.txt',
    'AdamW HP1':    './hyperparameters1/lstm_hp1_adamw.txt',
    'SGD HP1':      './hyperparameters1/lstm_hp1_sgd.txt'
}

hp2_files = {
    'Adadelta HP2':     './hyperparameters2/lstm_hp2_adadelta.txt',
    'AdamW HP2':        './hyperparameters2/lstm_hp2_adamw.txt',
    'SGD HP2':          './hyperparameters2/lstm_hp2_sgd.txt'
}

scheduler_files = {
    'Adadelta S':   './scheduler/lstm_scheduler_adadelta.txt',
    'AdamW S':      './scheduler/lstm_scheduler_adamw.txt',
    'SGD S':        './scheduler/lstm_scheduler_sgd.txt'
}

# Plotting training and validation losses
# plt.figure(figsize=(12, 8))

# for label, filepath in scheduler_files.items():
#     # Extract data
#     epochs, training_losses, validation_losses, f1_scores = extract_data(filepath)
    
#     # Plot training and validation loss
#     plt.plot(epochs, training_losses, label=f'{label} - Training Loss')
#     plt.plot(epochs, validation_losses, label=f'{label} - Validation Loss', linestyle='--')

# # Customize the plot
# plt.title('Sequencer-2d With Scheduler Loss', fontsize='20')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='upper left', prop={'size':15})
# plt.grid(True)
# plt.tight_layout()
# plt.ylim(0, 2)
# # Save the plot
# plt.savefig('./Visualization/lstm_scheduler_loss.png')
# plt.show()

# Plotting F1-scores
plt.figure(figsize=(12, 8))
cmap = plt.get_cmap('tab20')

i = 0
for label, filepath in files.items():
    # Extract data
    epochs, _, _, f1_scores = extract_data(filepath)
    
    # Plot F1-Score
    plt.plot(epochs, f1_scores, color=cmap(i), label=f'{label} - F1 Score')
    i += 1

# Customize the plot
plt.title('Sequencer-2d Validation F1-Score Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend(loc='lower right', prop={'size':10})
plt.grid(True)
plt.tight_layout()
plt.ylim(0, 1)
# Save the plot
plt.savefig('./Visualization/lstm_f1_scores.png')
plt.show()
