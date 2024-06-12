"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils,loss_and_accuracy_curve_plotter,testing
from torchvision import transforms
from timeit import default_timer as timer 

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
CONFIG_NAME = 50


#going modular/data/pizza_steak_sushi/train
# Setup directories
root_dir = "/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/results/persistence images/below 90th percentile/h1/npy_combined_features" # has 5 classes
annotation_file_path = "/Users/h6x/ORNL/git/modeling-ideas/overdose modeling for entire country/data/processed data/svi with hepvu/2018/annotation 2018/annotation.csv"


# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, validation_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    annotation_file_path=annotation_file_path,
    root_dir=root_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
# model = model_builder.SEResNeXt(CONFIG_NAME).to(device)
model = model_builder.SEResNet(CONFIG_NAME).to(device)


# Set loss and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# add a optimizer SGD
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)


# betas=(0.92, 0.999)

start_time = timer()

# Start training with help from engine.py
results = engine.train(model=model,
             train_dataloader=train_dataloader,
             validation_dataloader=validation_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             use_mixed_precision=True,
             save_name="se_restnet.pth",
             save_path="/Users/h6x/ORNL/git/persistence-image-classification/scratch model 1/models/")

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
print(f"Total training time: {(end_time-start_time)/60:.3f} minutes")

#plotting the results
loss_and_accuracy_curve_plotter.plot_loss_curves(results)

# Test the model after training
test_loss, test_acc = testing.test_step(model=model,
                                  dataloader=test_dataloader,
                                  loss_fn=loss_fn,
                                  device=device,
                                  use_mixed_precision=True)

# Print out test results
print(
    f"Test results | "
    f"test_loss: {test_loss:.4f} | "
    f"test_acc: {test_acc:.4f}"
)

# # Update results dictionary with test results
# results["test_loss"].append(test_loss)
# results["test_acc"].append(test_acc)

