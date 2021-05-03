# Datasets

## The general workflow

The framework adheres to the style of 
[Datasets](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) and 
[Dataloaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) 
as provided by PyTorch. For day to day use with DeePyMoD we provide the Dataset  and an easy dataloader `GPULoader`, To create
a train and test split we provide the tool `get_train_test_loader` all of these
can be a 