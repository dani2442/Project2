import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

import os
from datetime import datetime

from utils.model import AudioClassifier, ModelDimensions
from utils.dataset import NUM_CLASSES, get_dataset, SongDataset, SongDatasetTest
from utils.train import train, valid_epoch, calculate_confusion_matrix, plot_confusion_matrix
from sklearn.utils import class_weight

def main(l_rate, gamma, n_epochs, batch_size, save_path, seed):
    # replicability
    #torch.manual_seed(seed)

    # select device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = get_dataset('annotations_v3.csv')
    train_set, valid_set, test_set = random_split(dataset, [0.8,0.1,0.1])#, generator=torch.Generator().manual_seed(seed))

    # loss function
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=[0,1], y=[label for _, label in train_set])
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))

    # load train & validation loader
    train_set = SongDataset(train_set)
    valid_set = SongDatasetTest(valid_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=2)
    
    # select model
    #dims = ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=28, n_audio_head=2, n_audio_layer=2)
    dims = ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4)
    
    model = AudioClassifier(dims).to(device)
    model.load_state_dict(torch.load('models/pretrained.pth'))

    # logger
    log_path = os.path.join('log', datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    writer = SummaryWriter(log_path)

    # train
    train(model, train_loader, valid_loader, loss_fn, device, save_path, writer, lr=l_rate, n_epochs=n_epochs, gamma=gamma)

    # load best model
    model = AudioClassifier(dims).to(device)
    model.load_state_dict(torch.load(save_path))

    # test
    test_set = SongDatasetTest(test_set)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loss, test_accuracy = valid_epoch(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss}; Test accuracy: {test_accuracy}")

    # get confusion matrix
    m = calculate_confusion_matrix(model, test_loader, device)
    plot_confusion_matrix(m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma parameter for optimizer scheduler')   
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--save_path', type=str, default='models/best_model_transformer_v2.pth', help='best model saved path')
    parser.add_argument('--seed', type=int, default=1111)
    kwargs = parser.parse_args()

    main(**vars(kwargs))
