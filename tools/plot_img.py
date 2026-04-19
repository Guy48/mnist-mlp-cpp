import os
import sys
import numpy as np
import matplotlib.pyplot as plt

IMG_EDGE = 28
IMG_LEN = IMG_EDGE * IMG_EDGE


def main(argv):
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <image_file>")
        return 1

    image_path = argv[1]
    if not os.path.exists(image_path):
        print(f"Error: invalid path given: {image_path}")
        return 1

    img = np.fromfile(image_path, dtype=np.float32)
    if img.size != IMG_LEN:
        print(f"Error: expected {IMG_LEN} floats, got {img.size}")
        return 1

    plt.imshow(img.reshape(IMG_EDGE, IMG_EDGE), cmap="gray")
    plt.axis("off")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
