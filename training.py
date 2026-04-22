import torch
from metrics import calculate_psnr

# Function to save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"Checkpoint saved at {save_path}")

# Function to load checkpoint
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    loss = checkpoint['loss']
    print(f"Resumed from epoch {epoch}, loss {loss:.4f}")
    return model, optimizer, scheduler, epoch, loss


# Define the Training function
def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=10,
                         save_path="./weights/best_model.pth", checkpoint_path=None,
                         scheduler=None):
    best_val_loss = float('inf')  # Initialize with a very high value
    best_model_wts = None  # To store the best model weights
    start_epoch = 0  # Default start epoch
    loss = None  # Default loss
    if checkpoint_path:
        model, optimizer, scheduler, start_epoch, loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)


    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for batch_idx, (low_res_patches, high_res_patches) in enumerate(train_dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: Get predictions (low-res patches are input)
            outputs = model(low_res_patches)  # Forward pass through the SRCNN

            # Compute the loss (Mean Squared Error)
            loss = criterion(outputs, high_res_patches)
            running_loss += loss.item()

            # Backward pass: Compute gradients and update weights
            loss.backward()
            optimizer.step()

        # Print the average loss for the current epoch
        avg_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Optionally validate the model after every epoch
        if (epoch + 1) % 2 == 0:  # For example, validate every 2 epochs
            val_loss, val_psnr = validate(model, val_dataloader, criterion)

            # If validation loss improves, save the model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = model.state_dict()  # Save the model's state_dict
                print(f'Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}), saving model...')
                torch.save(best_model_wts, save_path)  # Save the best model

        # Optionally apply the learning rate scheduler after each epoch
        if scheduler:
            scheduler.step()

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, scheduler, epoch, running_loss, save_path ="weights/last_model.pth")



    # Load the best model weights (optional, but good practice to reload the best model)
    model.load_state_dict(best_model_wts)
    print(f"Best model saved at {save_path}")
    return model

# Define the Validation function
def validate(model, val_dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():  # No need to compute gradients for validation
        for low_res_patches, high_res_patches in val_dataloader:
            # Forward pass: Get predictions
            outputs = model(low_res_patches)

            # Compute the loss (Mean Squared Error)
            loss = criterion(outputs, high_res_patches)
            total_loss += loss.item()

            # Compute PSNR for each batch
            batch_psnr = 0.0
            for i in range(low_res_patches.size(0)):
                batch_psnr += calculate_psnr(high_res_patches[i].cpu().numpy(), outputs[i].cpu().numpy())

            total_psnr += batch_psnr / low_res_patches.size(0)

    # Compute the average validation loss and PSNR
    avg_val_loss = total_loss / len(val_dataloader)
    avg_psnr = total_psnr / len(val_dataloader)
    print(f'Validation Loss: {avg_val_loss:.4f}, Validation PSNR: {avg_psnr:.4f}')
    return avg_val_loss, avg_psnr






# Define the Training function for Conditional SRCNN
def condionnal_train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=10,
                     save_path="./weights/best_model.pth", checkpoint_path= None, scheduler=None):
    best_val_loss = float('inf')  # Initialize with a very high value
    best_model_wts = None  # To store the best model weights
    start_epoch = 0  # Default start epoch
    loss = None  # Default loss

    if checkpoint_path:
        model, optimizer, scheduler, start_epoch, loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)

    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for batch_idx, (low_res_patches, conditional_patches, high_res_patches) in enumerate(train_dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: Get predictions (low-res and conditional patches as input)
            outputs = model(low_res_patches, conditional_patches)  # Forward pass through the Conditional SRCNN

            # Compute the loss (Mean Squared Error)
            loss = criterion(outputs, high_res_patches)
            running_loss += loss.item()

            # Backward pass: Compute gradients and update weights
            loss.backward()
            optimizer.step()

        # Print the average loss for the current epoch
        avg_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Optionally validate the model after every epoch
        if (epoch + 1) % 2 == 0:  # For example, validate every 2 epochs
            val_loss, val_psnr = condionnal_validate(model, val_dataloader, criterion)

            # If validation loss improves, save the model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = model.state_dict()  # Save the model's state_dict
                print(f'Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}), saving model...')
                torch.save(best_model_wts, save_path)  # Save the best model

        # Optionally apply the learning rate scheduler after each epoch
        if scheduler:
            scheduler.step()  # Step after the epoch (for time-based schedulers like StepLR)

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, scheduler, epoch, running_loss, checkpoint_path)

    # Load the best model weights (optional, but good practice to reload the best model)
    model.load_state_dict(best_model_wts)
    print(f"Best model saved at {save_path}")
    return model



# Define the Validation function for Conditional SRCNN
def condionnal_validate(model, val_dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():  # No need to compute gradients for validation
        for low_res_patches, conditional_patches, high_res_patches in val_dataloader:
            # Forward pass: Get predictions
            outputs = model(low_res_patches, conditional_patches)
            print(low_res_patches.cpu().numpy().min(), low_res_patches.cpu().numpy().max())
            print(conditional_patches.cpu().numpy().min(), conditional_patches.cpu().numpy().max())
            print(high_res_patches.cpu().numpy().min(), high_res_patches.cpu().numpy().max())


            # Compute the loss (Mean Squared Error)
            loss = criterion(outputs, high_res_patches)
            total_loss += loss.item()

            # Compute PSNR for each batch
            batch_psnr = 0.0
            for i in range(low_res_patches.size(0)):
                batch_psnr += calculate_psnr(high_res_patches[i].cpu().numpy(), outputs[i].cpu().numpy())


            total_psnr += batch_psnr / low_res_patches.size(0)

    # Compute the average validation loss and PSNR
    avg_val_loss = total_loss / len(val_dataloader)
    avg_psnr = total_psnr / len(val_dataloader)
    print(f'Validation Loss: {avg_val_loss:.4f}, Validation PSNR: {avg_psnr:.4f}')
    return avg_val_loss, avg_psnr
