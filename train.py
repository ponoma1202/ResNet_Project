import torchvision
import torch
from torchvision import transforms
from tqdm import tqdm

from model import resnet18, resnet34

def main():
    # parameters
    num_epochs = 100
    num_classes = 10                # CIFAR-10 dataset
    learning_rate = 0.01

    # TODO: write custom dataset for the images Xiaofei provided
    train_transform = torchvision.transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.RandAugment(),
                                                      transforms.ToTensor(), 
                                                      transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
    val_transform = torchvision.transforms.Compose([transforms.ToTensor(), 
                                                      transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])       # totensor, normalize
    
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = resnet18(num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_accuracy, train_loss = train_epoch(train_loader, model, optimizer, criterion, device, epoch, num_epochs)
        val_accuracy, val_loss = validate_epoch(val_loader, model, optimizer, criterion, device)
        print(f'Training Loss: {train_loss}, Training Accuracy: {train_accuracy}')
        print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

def train_epoch(train_loader, model, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    total_correct = 0
    total_loss = 0

    for step, data in enumerate(tqdm(train_loader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        pred = outputs.squeeze().argmax(dim=1)
        total_loss += loss.item()
        total_correct += (pred == labels).sum().item() 
    
    accuracy = total_correct / len(train_loader.dataset)
    avg_loss = total_loss/ len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, Accuracy: {accuracy}')
    return accuracy, avg_loss
          
def validate_epoch(val_loader, model, optimizer, criterion, device):
    model.eval()
    print("Starting validation...")

    with torch.no_grad():
        total_correct = 0
        total_loss = 0

        for step, data in tqdm(enumerate(val_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            pred = outputs.squeeze().argmax(dim=1)
            total_loss += loss.item()
            total_correct += (pred == labels).sum().item()

        accuracy = total_correct / len(val_loader.dataset)
        avg_loss = total_loss / len(val_loader.dataset)
        return accuracy, avg_loss


if __name__ == "__main__":
    main()