import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

import os
from datetime import datetime

from utils.model import AudioClassifier, ModelDimensions
from utils.dataset import NUM_CLASSES, get_dataset, SongDataset_v2, SongDatasetTest_v2
from utils.train import train, valid_epoch, calculate_confusion_matrix, plot_confusion_matrix

def main(l_rate, gamma, n_epochs, batch_size, save_path, seed):
    # replicability
    torch.manual_seed(seed)

    # select device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # load dataset
    dataset = get_dataset('annotations_v2.csv')
    train_set, valid_set, test_set = random_split(dataset, [0.85,0.05,0.1], generator=torch.Generator().manual_seed(seed))

    # load train & validation loader
    train_set = SongDataset_v2(train_set)
    valid_set = SongDatasetTest_v2(valid_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=2)
    
    # select model
    dims = ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4)
    
    model = AudioClassifier(dims).to(device)
    # model.load_state_dict(torch.load('models/pretrained.pth'))

    # logger
    log_path = os.path.join('log', datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    writer = SummaryWriter(log_path)

    # train
    train(model, train_loader, valid_loader, loss_fn, device, save_path, writer, lr=l_rate, n_epochs=n_epochs, gamma=gamma)

    # load best model
    model = AudioClassifier(dims).to(device)
    model.load_state_dict(torch.load(save_path))

    # test
    test_set = SongDatasetTest_v2(test_set)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loss, test_accuracy = valid_epoch(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss}; Test accuracy: {test_accuracy}")

    # get confusion matrix
    m = calculate_confusion_matrix(model, test_loader, device)
    plot_confusion_matrix(m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='gamma parameter for optimizer scheduler')   
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--save_path', type=str, default='models/best_model_transformer.pth', help='best model saved path')
    parser.add_argument('--seed', type=int, default=1234)
    kwargs = parser.parse_args()

    main(**vars(kwargs))
