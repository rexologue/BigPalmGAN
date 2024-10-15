import argparse
import os
import torch

from train_loop import train
from model_utils import setup_generator, setup_discriminator
from data import setup_loader 
from support_utils import create_unique_directory
from fid.inception import setup_inception

if __name__ == '__main__':
    # Создание парсера аргументов
    parser = argparse.ArgumentParser(description="Setup parameters for BigGAN fine-tuning")

    # Добавление аргументов
    parser.add_argument('--epochs', type=int, required=True, help='Num of epochs for fine-tuning')
    parser.add_argument('--num_classes', type=int, required=True, help='Amount of classes')
    parser.add_argument('--lr_g', type=float, required=True, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, required=True, help='Discriminator learning rate')
    parser.add_argument('--weight_decay', type=float, required=True, help='Weight decay')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--adversarial_loss_weight', type=float, required=True, help='Adversarial loss weight')
    parser.add_argument('--pixel_loss_weight', type=float, required=True, help='Pixel loss weight')
    parser.add_argument('--classification_loss_weight', type=float, required=True, help='Classification loss weight')
    parser.add_argument('--accumulation_steps', type=int, required=True, help='Steps for gradient accumulation')
    parser.add_argument('--unfreeze_last_n', type=int, required=True, help='How much last generator layers to unfreeze')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to train directory')
    parser.add_argument('--eval_dir', type=str, required=True, help='Path to eval directory')
    parser.add_argument('--print_every_n_batches', type=int, required=True, help='How often to print training stats')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir')
    parser.add_argument('--inception', type=str, required=True, help='Path to the inception model')
    parser.add_argument('--ckp', type=str, required=False, default=None, help='Path to checkpoint')

    # Парсинг аргументов
    args = parser.parse_args()
    
    if args.ckp is not None:
        checkpoint = torch.load(args.ckp)
    else:
        checkpoint = None
        
    generator = setup_generator(args.num_classes, args.unfreeze_last_n, checkpoint)
    discriminator = setup_discriminator(args.num_classes, checkpoint)
    inception = setup_inception(args.inception)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    generator.to(device)
    discriminator.to(device)
    inception.to(device)
        
    optimizer_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_g, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999), weight_decay=args.weight_decay)   
        
    if args.ckp is not None:
        if checkpoint['optimizer_G_state_dict'] is not None: 
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        if checkpoint['optimizer_G_state_dict'] is not None:    
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
    train_loader = setup_loader(args.train_dir, args.batch_size)
    eval_loader = setup_loader(args.eval_dir, args.batch_size, train_phase=False)
    
    adversarial_loss = torch.nn.BCEWithLogitsLoss()
    pixel_loss = torch.nn.MSELoss()
    classification_loss = torch.nn.CrossEntropyLoss()
    
    run_dir = create_unique_directory(os.path.join(args.output_dir, 'biggan_fine_tune_run'))
    
    w_dir = os.path.join(run_dir, 'weights')
    os.makedirs(w_dir, exist_ok=True)

    img_dir = os.path.join(run_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    start_epoch = -1 if checkpoint is None else checkpoint['epoch']
    
    train(
        epochs=args.epochs,
        start_epoch=start_epoch,
        generator=generator,
        discriminator=discriminator,
        inception=inception,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        train_loader=train_loader,
        eval_loader=eval_loader,
        adversarial_loss=adversarial_loss,
        pixel_loss=pixel_loss,
        classification_loss=classification_loss,
        adversarial_loss_weight=args.adversarial_loss_weight,
        pixel_loss_weight=args.pixel_loss_weight,
        classification_loss_weight=args.classification_loss_weight,
        accumulation_steps=args.accumulation_steps,
        print_every_n_batches=args.print_every_n_batches,
        num_classes=args.num_classes,
        img_dir=img_dir,
        w_dir=w_dir,
        device=device
    )