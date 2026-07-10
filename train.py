import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from model import UNet
from save import save
from dataset import train_loader, validation_loader
from dice import dice
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
loss_fn = nn.BCEWithLogitsLoss()   
optimizer = optim.Adam(model.parameters(),lr = 1e-4)
best =0 
total=0
num_epochs=10
for epoch in range(num_epochs):

    model.train()
    total_loss=0
    train_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for images,masks in train_bar:
        print(f"running epoch {epoch+1}")
        images = images.to(device)
        masks = masks.to(device)
        if masks.size(1) == 3:
            masks = masks[:, 0:1, :, :]
        predictions = model(images)
        loss = loss_fn(predictions,masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss {loss.item():.4f}")
    model.eval()
    total=0
    val_bar = tqdm.tqdm(validation_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validate]")
    for images,masks in val_bar:
        images = images.to(device)
        masks = masks.to(device)
        if masks.size(1) == 3:
            masks = masks[:, 0:1, :, :]
        with torch.no_grad():
            predictions = model(images)
            dice_score = dice(predictions,masks)
            total += dice_score.item()
            loss = loss_fn(predictions,masks)
        print(f"Validation Loss: {loss.item()}")
    val = total/len(validation_loader)
    loss_avg=total_loss/len(train_loader)
    
    save(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        filename=f"checkpoint_{epoch}.pt"
    )
    if val>best:
        best=val
        save(model,optimizer,epoch,filename="best.pt")
