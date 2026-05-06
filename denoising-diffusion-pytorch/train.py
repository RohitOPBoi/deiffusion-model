import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def main():

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Running on CPU")

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        flash_attn=False
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=100,
        objective='pred_v'
    )

    trainer = Trainer(
        diffusion,
        "D:/the one and only/denoising-diffusion-pytorch/data/oct",

        train_batch_size=4,
        gradient_accumulate_every=4,

        train_lr=2e-4,
        train_num_steps=12000,

        ema_decay=0.999,
        amp=True,

        calculate_fid= False,

        save_and_sample_every=2000,
        results_folder='./results'
    )

    trainer.train()
    print("Training complete. Samples in ./results/")

if __name__ == "__main__":
    main()