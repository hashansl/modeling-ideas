"""
Contains functions for testing a PyTorch model.
"""
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt


from tqdm.auto import tqdm
from typing import Dict, List, Tuple

# def test_step(model: torch.nn.Module, 
#               dataloader: torch.utils.data.DataLoader, 
#               loss_fn: torch.nn.Module,
#               device: torch.device,
#               use_mixed_precision: bool = False) -> float:
#     """Tests a PyTorch model.

#     Turns a target PyTorch model to "eval" mode and then performs
#     a forward pass on a testing dataset.

#     Args:
#         model: A PyTorch model to be tested.
#         dataloader: A DataLoader instance for the model to be tested on.
#         loss_fn: A PyTorch loss function to calculate loss on the test data.
#         device: A target device to compute on (e.g. "cuda" or "cpu").

#     Returns:
#         A tuple of testing loss and testing accuracy metrics.
#         In the form (test_loss, test_accuracy). For example:

#         (0.0223, 0.8985)
#     """
#     # Put model in eval mode
#     model.eval()

#     # Setup test loss and test accuracy values
#     test_loss = 0

#     # Turn on inference context manager
#     with torch.inference_mode():  # similar to torch.no_grad()
#         with torch.cuda.amp.autocast(enabled=use_mixed_precision):
#             # Loop through DataLoader batches
#             for batch, (X, y) in enumerate(dataloader):
#                 # Send data to target device
#                 X, y = X.to(device), y.to(device)

#                 # Reshape target tensor to match model output shape
#                 y = y.view(-1, 1)

#                 # 1. Forward pass
#                 test_pred_logits = model(X)

#                 # 2. Calculate and accumulate loss
#                 loss = loss_fn(test_pred_logits, y)
#                 test_loss += loss.detach().cpu().item()


#     # Adjust metrics to get average loss and accuracy per batch 
#     test_loss = test_loss / len(dataloader)


#     plt.figure()
#     plt.plot(y_pred.detach().numpy(), label='predicted')
#     plt.plot(y_train.numpy(), label='actual')
#     plt.ylabel('output y')
#     plt.legend()
#     plt.show()

#     return test_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              use_mixed_precision: bool = False) -> float:
    """Tests a PyTorch model.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        use_mixed_precision: A boolean flag to use mixed precision during testing.

    Returns:
        A float representing the average test loss.
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss
    test_loss = 0

    # Lists to store actual and predicted values for plotting
    all_preds = []
    all_labels = []

    # Turn on inference context manager
    with torch.inference_mode():  # similar to torch.no_grad()
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            # Loop through DataLoader batches
            for batch, (X, y) in enumerate(dataloader):
                # Send data to target device
                X, y = X.to(device), y.to(device)

                # Reshape target tensor to match model output shape
                y = y.view(-1, 1)

                # 1. Forward pass
                test_pred_logits = model(X)

                print("Predicted: ",test_pred_logits)
                print("Actual: ",y)

                # 2. Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.detach().cpu().item()

                # Collect predictions and labels
                all_preds.append(test_pred_logits.detach().cpu().numpy())
                all_labels.append(y.detach().cpu().numpy())

    # Adjust metrics to get average loss
    test_loss = test_loss / len(dataloader)

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Flatten the arrays if necessary
    all_preds = all_preds.flatten()
    all_labels = all_labels.flatten()

    # create index from length of all_preds
    index = np.arange(len(all_preds))

    # Plot actual vs predicted
    plt.figure()
    # plot scatter plot
    plt.scatter(index, all_labels, c="g", s=4, label="Actual", alpha=0.5)
    plt.scatter(index, all_preds, c="r", s=4, label="Predicted", alpha=0.5)

    # plt.plot(all_labels, label='Actual')
    # plt.plot(all_preds, label='Predicted')
    plt.ylabel('Output y')
    # plt.legend()
    # plt.savefig('/home/h6x/Projects/overdose_modeling/SEResNet_regression/plots/actual_vs_predicted_1.png')
    plt.show()

    return test_loss
