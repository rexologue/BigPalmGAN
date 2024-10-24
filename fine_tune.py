import argparse
import os
import torch
from train_loop import train
from model_utils import setup_generator, setup_discriminator
from data import setup_loader
from support_utils import create_unique_directory, str2bool
from fid.inception import setup_inception
from losses import InceptionPerceptualLoss, DiscriminatorHingeLoss, GeneratorHingeLoss

if __name__ == '__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Setup parameters for BigGAN fine-tuning")
    
    parser.add_argument('--epochs', type=int, required=True, help='Num of epochs for fine-tuning')
    parser.add_argument('--num_classes', type=int, required=True, help='Amount of classes')
    parser.add_argument('--lr_g', type=float, required=True, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, required=True, help='Discriminator learning rate')
    parser.add_argument('--weight_decay_g', type=float, required=True, help='Weight decay of generator')
    parser.add_argument('--weight_decay_d', type=float, required=True, help='Weight decay of discriminator')
    parser.add_argument('--betas_g', nargs='+', type=float, required=True, help='Betas for generator optimizer')
    parser.add_argument('--betas_d', nargs='+', type=float, required=True, help='Betas for discriminator optimizer')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--g_loss_weight', type=float, required=True, help='Generator loss function weight')
    parser.add_argument('--pixel_loss_weight', type=float, required=True, help='Pixel loss weight')
    parser.add_argument('--perceptual_loss_weight', type=float, required=True, help='Perceptual loss weight')
    parser.add_argument('--accumulation_steps', type=int, required=True, help='Steps for gradient accumulation')
    parser.add_argument('--lambda_gp', type=float, required=True, help='Lambda coefficient of gradient penalty')
    parser.add_argument('--unfreeze_last_n', type=int, required=True, help='How much last generator layers to unfreeze')
    parser.add_argument('--use_augs', type=str2bool, nargs='?', const=True, default=False, help='Use augmentations or not')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to train directory')
    parser.add_argument('--eval_dir', type=str, required=True, help='Path to eval directory')
    parser.add_argument('--print_every_n_batches', type=int, required=True, help='How often to print training stats')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir')
    parser.add_argument('--inception', type=str, required=True, help='Path to the inception model')
    parser.add_argument('--ckp', type=str, required=False, default=None, help='Path to checkpoint')
    parser.add_argument('--only_g_ckp', type=str, required=False, default=None, help='Path to checkpoint, where saved only generator weights')
    
    args = parser.parse_args()

    if args.ckp is not None:
        checkpoint = torch.load(args.ckp)
    else:
        checkpoint = dict()

    if args.only_g_ckp is not None:
        g_ckp = torch.load(args.only_g_ckp)
    else:
        g_ckp = checkpoint.get('generator_state_dict', None)

    d_ckp = checkpoint.get('discriminator_state_dict', None)
    g_opt_ckp = checkpoint.get('optimizer_G_state_dict', None)
    d_opt_ckp = checkpoint.get('optimizer_D_state_dict', None)
    ckp_start_epoch = checkpoint.get('epoch', None)

    generator = setup_generator(args.num_classes, args.unfreeze_last_n, g_ckp)
    discriminator = setup_discriminator(args.num_classes, d_ckp)
    inception = setup_inception(args.inception)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generator.to(device)
    discriminator.to(device)
    inception.to(device)

    optimizer_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, generator.parameters()),
                                    lr=args.lr_g, betas=tuple(args.betas_g), weight_decay=args.weight_decay_g)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(),
                                    lr=args.lr_d, betas=tuple(args.betas_d), weight_decay=args.weight_decay_d)

    if g_opt_ckp is not None:
        optimizer_G.load_state_dict(g_opt_ckp)
    if d_opt_ckp is not None:
        optimizer_D.load_state_dict(d_opt_ckp)

    train_loader = setup_loader(args.train_dir, args.batch_size, args.use_augs)
    eval_loader = setup_loader(args.eval_dir, args.batch_size, args.use_augs, train_phase=False)

    g_loss_fn = GeneratorHingeLoss()
    d_loss_fn = DiscriminatorHingeLoss()
    pixel_loss = torch.nn.MSELoss()
    perceptual_loss = InceptionPerceptualLoss()

    run_dir = create_unique_directory(os.path.join(args.output_dir, 'biggan_fine_tune_run'))

    w_dir = os.path.join(run_dir, 'weights')
    os.makedirs(w_dir, exist_ok=True)

    img_dir = os.path.join(run_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    start_epoch = -1 if ckp_start_epoch is None else ckp_start_epoch

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
        g_loss_fn=g_loss_fn,
        d_loss_fn=d_loss_fn,
        pixel_loss=pixel_loss,
        perceptual_loss=perceptual_loss,
        g_loss_weight=args.g_loss_weight,
        pixel_loss_weight=args.pixel_loss_weight,
        perceptual_loss_weight=args.perceptual_loss_weight,
        accumulation_steps=args.accumulation_steps,
        lambda_gp=args.lambda_gp,
        print_every_n_batches=args.print_every_n_batches,
        num_classes=args.num_classes,
        img_dir=img_dir,
        w_dir=w_dir,
        device=device
    )
