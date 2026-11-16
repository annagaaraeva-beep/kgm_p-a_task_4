import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw

points = np.loadtxt("DS8.txt", dtype=int)

mask = np.zeros((540, 960), dtype=bool)
mask[539 - points[:, 1], points[:, 0]] = True

regions, n = label(mask, np.ones((3, 3)))
centers = np.array(center_of_mass(mask, regions, range(1, n + 1)))[:, ::-1]

y, x = np.indices(mask.shape)
nearest = cKDTree(centers).query(np.c_[x.ravel(), y.ravel()])[1].reshape(mask.shape)

pixels = np.full((540, 960, 3), 255, dtype=np.uint8)
pixels[mask] = 230
pixels[1:][nearest[1:] != nearest[:-1]] = (0, 0, 255)
pixels[:, 1:][nearest[:, 1:] != nearest[:, :-1]] = (0, 0, 255)

image = Image.fromarray(pixels)
draw = ImageDraw.Draw(image)

for x, y in centers:
    x, y = round(x), round(y)
    draw.ellipse((x-2, y-2, x+2, y+2), fill="red")

image.save("result.png")