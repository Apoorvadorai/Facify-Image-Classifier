import argparse
import os
from datetime import datetime

import torch
import torchvision.transforms as transforms
from dataset_vgg import FaceDataset, AttributesDataset, mean, std
from model_vgg import MultiOutputModel
from test import calculate_metrics, validate, visualize_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from test import checkpoint_load


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--attributes_file', type=str, default='data/UTKFace/newattributes.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to the checkpoint")

    args = parser.parse_args()

    start_epoch = 1
    N_epochs = 50
    batch_size = 32
    lr = 0.00001
    num_workers = 8 # number of processes to handle dataset loading
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    model = MultiOutputModel(n_age_classes=attributes.num_age,
                             n_gender_classes=attributes.num_gender,
                             n_ethnicity_classes=attributes.num_ethnicity).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # specify image transforms for augmentation during training
    train_transform = transforms.Compose([
        transforms.Resize((80,80)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2),
                                # shear=None, resample=False, fillcolor=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = FaceDataset('data/UTKFace/newtrain.csv', attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = FaceDataset('data/UTKFace/newval.csv', attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    
    if args.checkpoint is not None:
        start_epoch = checkpoint_load(model, args.checkpoint)

    
    
    

    logdir = os.path.join('./logs/expUTK_vgg', get_cur_time())
    savedir = os.path.join('./checkpoints/expUTK_vgg', get_cur_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataloader)

    # Uncomment rows below to see example images with ground truth labels in val dataset and all the labels:
    # visualize_grid(model, train_dataloader, attributes, device, show_cn_matrices=False, show_images=True, checkpoint=None, show_gt=True)
    # print("\nAll gender labels:\n", attributes.eyes_labels)
    # print("\nAll age labels:\n", attributes.age_labels)
    # print("\nAll ethnicity labels:\n", attributes.ethnicity_labels)

    print("Starting training ...")

    for epoch in range(start_epoch, N_epochs + 1):
        total_loss = 0
        accuracy_age = 0
        accuracy_gender = 0
        accuracy_ethnicity = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
    
            batch_accuracy_age, batch_accuracy_gender, batch_accuracy_ethnicity = \
                calculate_metrics(output, target_labels)

            accuracy_age += batch_accuracy_age
            accuracy_gender += batch_accuracy_gender
            accuracy_ethnicity += batch_accuracy_ethnicity

            loss_train.backward()
            optimizer.step()

        print("epoch {:4d}, loss: {:.4f}, age: {:.4f}, gender: {:.4f}, ethnicity: {:.4f}".format(
            epoch,
            total_loss / n_train_samples,
            accuracy_age / n_train_samples,
            accuracy_gender / n_train_samples,
            accuracy_ethnicity / n_train_samples))

        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

        if epoch % 5 == 0:
            #validate model
            visualize_grid(model, val_dataloader, attributes, device, show_cn_matrices=False, show_images=False, checkpoint=None, show_gt=False)


        if epoch % 10 == 0:
            #save model weights
            checkpoint_save(model, savedir, epoch)
