import torch

# Train the network
def train_network(epochs, steps, train_loader, test_loader, model, device, optimizer, criterion):
    """Train the network

    Parameters:
    epochs (int): epochs number
    steps  (int): the number of steps
    train_loader  (list): the train loader images
    test_loader (list): the test loader images
    model (object): the trained model
    device (string): device (cpu or cuda)
    optimizer (object): training optimizer
    criterion (object): triing criterion
   """
    
    print()
    print("Training the network...")
    print()
    
    train_losses, test_losses = [], []
    running_loss = 0
    print_every = 10
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss/len(train_loader))
                test_losses.append(test_loss/len(test_loader))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(test_loader):.3f}.. "
                      f"Test accuracy: {accuracy/len(test_loader):.3f}")
                running_loss = 0
                model.train()