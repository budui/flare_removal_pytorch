from PIL import Image
from torchvision.transforms.functional import to_tensor


def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert("RGB")
    if size is not None:
        size = (size, size) if isinstance(size, int) else size
        img = img.resize(size, Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize(
            (int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS
        )
    return to_tensor(img).unsqueeze_(0)


def save_image(filename, data):
    img = data.mul(255.0).clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)
