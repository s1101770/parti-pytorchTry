import torch
from parti_pytorch import Parti, VitVQGanVAE

# first instantiate your ViT VQGan VAE
# a VQGan VAE made of transformers

vit_vae = VitVQGanVAE(
    dim = 256,               # dimensions
    image_size = 256,        # target image size
    patch_size = 16,         # size of the patches in the image attending to each other
    num_layers = 3           # number of layers
).cuda()

vit_vae.load_state_dict(torch.load(f'/path/to/vae.pt')) # you will want to load the exponentially moving averaged VAE

# then you plugin the ViT VqGan VAE into your Parti as so

parti = Parti(
    vae = vit_vae,            # vit vqgan vae
    dim = 512,                # model dimension
    depth = 8,                # depth
    dim_head = 64,            # attention head dimension
    heads = 8,                # attention heads
    dropout = 0.,             # dropout
    cond_drop_prob = 0.25,    # conditional dropout, for classifier free guidance
    ff_mult = 4,              # feedforward expansion factor
    t5_name = 't5-large',     # name of your T5
)

# ready your training text and images

texts = [
    'a child screaming at finding a worm within a half-eaten apple',
    'lizard running across the desert on two feet',
    'waking up to a psychedelic landscape',
    'seashells sparkling in the shallow waters'
]

images = torch.randn(4, 3, 256, 256).cuda()

# feed it into your parti instance, with return_loss set to True

loss = parti(
    texts = texts,
    images = images,
    return_loss = True
)

loss.backward()

# do this for a long time on much data
# then...

images = parti.generate(texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
], cond_scale = 3., return_pil_images = True) # conditioning scale for classifier free guidance

# List[PILImages] (256 x 256 RGB)