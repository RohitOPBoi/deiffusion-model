import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from pathlib import Path
import argparse

from denoising_diffusion_pytorch import Unet, GaussianDiffusion


def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2


def load_image(path, size):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = Image.open(path).convert("L")
    return transform(img).unsqueeze(0)


def unwrap(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, default="./results/model-6.pt")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--start_t", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="./refine_outputs")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading checkpoint: {args.ckpt_path}")
    print(f"[INFO] Input image: {args.image_path}")

    Path(args.out_dir).mkdir(exist_ok=True)

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=1000,
        sampling_timesteps=100,
        objective='pred_v'
    ).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    diffusion.load_state_dict(ckpt["model"])

    diffusion.eval()

    x0 = load_image(args.image_path, args.image_size).to(device)

    t = torch.full((1,), args.start_t, device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x = diffusion.q_sample(x0, t, noise=noise)

    noisy = x.clone()

    with torch.no_grad():
        for i in reversed(range(1, args.start_t + 1)):
            

            out = diffusion.p_sample(x, i)
            x = out if not isinstance(out, tuple) else out[0]

    refined = x.detach().cpu()

    save_image(denorm(x0.cpu()), f"{args.out_dir}/original.png")
    save_image(denorm(noisy.cpu()), f"{args.out_dir}/noisy.png")
    save_image(denorm(refined), f"{args.out_dir}/refined.png")

    grid = make_grid(torch.cat([x0.cpu(), noisy.cpu(), refined], dim=0), nrow=3)
    save_image(denorm(grid), f"{args.out_dir}/comparison.png")

    print("Done!")
    print(f" Outputs saved in: {args.out_dir}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()