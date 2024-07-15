import context
import torch
import torch.nn as nn
import csv
import random

from model.transformer import Transformer
from train_test.dataset import get_dataloader, DataCtg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.load('transformer_model.pth')
model = model.to(device)

test_dataloader, _, _ = get_dataloader(DataCtg.TEST, 16)
test_data = list(test_dataloader)

flattened_test_data = []
for batch in test_data:
    labels, tokens = batch
    for i in range(labels.size(0)):
        flattened_test_data.append((labels[i], tokens[i]))

random.shuffle(flattened_test_data)

# Select the first 30 sequences
selected_data = flattened_test_data[:30]

criterion = nn.CrossEntropyLoss(ignore_index=24)

def calculate_accuracy(predictions, targets):
    _, preds = torch.max(predictions, dim=-1)
    correct = (preds == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()

# Open a CSV file to write the results
with open('random_test_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Input Tokens', 'True Labels', 'Predictions'])

    # Evaluate the model on the selected data and write predictions to CSV
    model.eval()
    test_running_loss, test_running_corrects, test_total = 0.0, 0, 0
    with torch.no_grad():
        for label, token in selected_data:
            token, label = token.unsqueeze(0).to(device), label.unsqueeze(0).to(device, dtype=torch.long)
            out = model(token, label)
            loss = criterion(out.view(-1, out.size(-1)), label.view(-1))
            test_running_loss += loss.item() * token.size(0)
            test_running_corrects += calculate_accuracy(out, label) * token.size(0)
            test_total += token.size(0)

            # Get the predictions
            _, preds = torch.max(out, dim=-1)
            
            # Write the data and predictions to the CSV file
            writer.writerow([
                token.cpu().numpy().tolist(), 
                label.cpu().numpy().tolist(), 
                preds.cpu().numpy().tolist()
            ])

test_loss = test_running_loss / test_total
test_acc = test_running_corrects / test_total

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
