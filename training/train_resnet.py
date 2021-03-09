import time
import copy
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from preprocessing import load_data
from nets.resnet import ResNet


def train_model(dataset_sizes, loader, model, criterion, optimizer, scheduler, device, num_epochs=100):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['training', 'val']:
            if phase == 'training':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            for inputs, labels in loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'val':
                scheduler.step(epoch_loss)

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(best_model_wts)
    return model


def train_resnet(data_dir, num_imgs_in_tif=1,
                 expand_data=False, patch_shape=(256, 256),
                 test_size=0.1, augment=True,
                 batch_size=16, epochs=100, lr=0.0004,
                 model_name='resnet.pt', basedir='save_models/'
                 ):
    if (not model_name.endswith('.pt')) or (not model_name.endswith('.pth')):
        model_name = model_name + '.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_sizes, loader = load_data(data_dir, num_imgs_in_tif, expand_data,
                                      test_size, patch_shape, batch_size, augment=augment, device=device)
    model = ResNet()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    model = train_model(dataset_sizes, loader, model, criterion, optimizer, scheduler, device=device, num_epochs=epochs)
    torch.save(model.state_dict(), basedir+model_name)
    return
