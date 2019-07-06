
import torch

def test_network(loader, device_test, model, criterion):
    print()
    print("Testing trained network accuracy")
    print()
    
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device_test), labels.to(device_test)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print("Test Loss: {:.3f}.. ".format(test_loss/len(loader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(loader)))