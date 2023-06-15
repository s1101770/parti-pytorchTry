import torch

from parti_pytorch import VitVQGanVAE, VQGanVAETrainer

vit_vae = VitVQGanVAE(
    dim = 256,               # dimensions
    image_size = 256,        # target image size
    patch_size = 16,         # size of the patches in the image attending to each other
    num_layers = 3           # number of layers
).cuda()

trainer = VQGanVAETrainer(
    vit_vae,
    folder = './TrainPicture',
    num_train_steps = 1000,
    lr = 3e-4,
    batch_size = 4,
    grad_accum_every = 8,
    amp = True
)

trainer.train().requires_grad_()