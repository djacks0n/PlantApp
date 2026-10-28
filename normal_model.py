import time
import csv
import os
import copy
import shutil
import torch
import torch.nn as nn
from torchvision import models

# old
# from dataset import get_dataloaders, NUM_CLASSES
from dataset import get_dataloaders, NUM_CLASSES, CLASS_NAMES, CONFUSABLE_PAIRS

seed = 67
frozen_epochs = 5
# old
# finetune_epochs = 10
finetune_epochs = 20
batch_size = 32
frozen_lr = 0.001 #drop to 0.0001 if too fast
finetune_lr = 0.00001

weights_save_path = "normal_model.pth" #save model
local_save_path = f"/tmp/{os.environ.get('USER', 'me')}_normal_model.pth" #server's own disk, not the home folder
accuracy_save_path = "normal_model.csv"
confusion_save_path = "normal_confusion_matrix.csv"
pairs_save_path = "normal_pair_results.csv"
# old
# resume = False #set True to continue after a crash

def make_model():
    model = models.resnet50(weights="IMAGENET1K_V2")

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def set_phase(model, epoch):
    if epoch <= frozen_epochs:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        return torch.optim.Adam(model.fc.parameters(), lr=frozen_lr), "frozen"

    for param in model.parameters():
        param.requires_grad = True   # unfreeze the hidden layer backbone
    return torch.optim.Adam(model.parameters(), lr=finetune_lr), "finetune"

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(dataloader)
    start = time.time()

    for batch_num, (images, labels) in enumerate(dataloader, start=1): #images and their correpsonding labels is first batch, start at batch num 1

        images = images.to(device)
        #print(f"Images tensor {images}")
        labels = labels.to(device)
        #print(f"Labels: {labels}")

        optimizer.zero_grad()
        outputs = model(images) # forward pass to get the class probabilities
        loss = criterion(outputs, labels) #use cross entropy loss with array of probabilities
        loss.backward() #backprop to create error gradients
        optimizer.step() #update the weights

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        #update every 1600 images out of the 19380
        if batch_num % 50 == 0 or batch_num == num_batches:
            elapsed_time = time.time() - start
            remaining = (elapsed_time/batch_num) * (num_batches - batch_num)
            print(f" batch {batch_num}/{num_batches} | "
                  f"loss {total_loss / total:.4f} | "
                  f"{(elapsed_time/batch_num):.2f}s per batch | "
                  f"about {remaining / 60:.1f} min left")

    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

def get_confusion_matrix(model, loader, device):
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long) #rows = true species, columns = predicted species

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            predictions = model(images).argmax(dim=1).cpu()
            for true_label, predicted_label in zip(labels, predictions):
                confusion[true_label, predicted_label] += 1

    return confusion

def save_confusion_matrix(confusion):
    with open(confusion_save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ predicted"] + CLASS_NAMES)
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name] + confusion[i].tolist())

def save_pair_results(confusion):
    rows = []
    for invasive, native in CONFUSABLE_PAIRS:
        invasive_total = max(confusion[invasive].sum().item(), 1)
        native_total = max(confusion[native].sum().item(), 1)
        invasive_called_native = confusion[invasive, native].item()
        native_called_invasive = confusion[native, invasive].item()

        rows.append({
            "invasive": CLASS_NAMES[invasive],
            "native": CLASS_NAMES[native],
            "invasive_acc": round(confusion[invasive, invasive].item() / invasive_total, 4),
            "native_acc": round(confusion[native, native].item() / native_total, 4),
            "invasive_called_native": invasive_called_native,
            "native_called_invasive": native_called_invasive,
            "pair_confusion_rate": round((invasive_called_native + native_called_invasive) / (invasive_total + native_total), 4),
        })

    with open(pairs_save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return rows

def main():
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("testing that saving works...")
    torch.save(torch.zeros(25_000_000), "save_test.pth")
    os.remove("save_test.pth")
    print("saving works")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = make_model().to(device) #load line under if crash
    #model.load_state_dict(torch.load(weights_save_path, map_location=device))

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = []
    start_epoch = 1
    optimizer, phase = set_phase(model, 1)
    best_weights = None
    run_start = time.time()

    # for epoch in range(1, (frozen_epochs + finetune_epochs) + 1):
    for epoch in range(start_epoch, (frozen_epochs + finetune_epochs) + 1):
        if epoch == 1:
            print(f"\ntraining only the last layer ({frozen_epochs} epochs)")
        if epoch == frozen_epochs + 1 and phase == "frozen":
            print(f"\n start the finetuning ({finetune_epochs} epochs)")
            optimizer, phase = set_phase(model, epoch)

        print(f"\nEpoch {epoch}/{(frozen_epochs+finetune_epochs)} ({phase})")
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_minutes = (time.time() - epoch_start) / 60
        print(f"train loss {train_loss:.4f} & train acc {train_acc:.4f}")
        print(f"val loss   {val_loss:.4f} & val acc   {val_acc:.4f} | {epoch_minutes:.1f} min")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()} #copy to regular memory, off the GPU
            print("  new best val accuracy")

        history.append({
            "epoch": epoch, "phase": phase,
            "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
        })

        with open(accuracy_save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    # load best model for a final test
    print(f"\nTraining done in {(time.time() - run_start) / 3600:.2f} hours.")
    model.load_state_dict(best_weights)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy:     {test_acc:.4f}")

    confusion = get_confusion_matrix(model, test_loader, device)
    save_confusion_matrix(confusion)
    pair_rows = save_pair_results(confusion)
    print(f"\nlook-alike pairs on the test set (saved to {pairs_save_path}):")
    for row in pair_rows:
        print(f"  {row['invasive']} vs {row['native']}: "
              f"{row['invasive_called_native']} invasive called native, "
              f"{row['native_called_invasive']} native called invasive, "
              f"confusion rate {row['pair_confusion_rate']:.4f}")

    print(f"\nsaving to {local_save_path}...")
    torch.save(best_weights, local_save_path)
    print("saved")
    print(f"copying to {weights_save_path}... if this hangs, Ctrl+C and copy it yourself from a new terminal:")
    print(f"  cp {local_save_path} {os.path.abspath(weights_save_path)}")
    shutil.copy(local_save_path, weights_save_path)
    print("copied")

if __name__ == "__main__":
    main()