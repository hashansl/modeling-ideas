import matplotlib.pyplot as plt
from typing import Dict, List

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "validation_loss": [...],
             "validation_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and validation)
    loss = results['train_loss']
    validation_loss = results['validation_loss']

    # Get the accuracy values of the results dictionary (training and validation)
    # accuracy = results['train_acc']
    # validation_accuracy = results['validation_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 1, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, validation_loss, label='validation_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    # plt.savefig('/home/h6x/Projects/overdose_modeling/SEResNet_regression/plots/loss_curve_test__2.png')
    plt.show()