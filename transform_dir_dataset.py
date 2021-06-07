import os, cv2
import numpy as np
from tqdm import tqdm


UNDIST_COORDS = np.load('/home/ntsuranov/mac/ReID/LensFunUndistCoords_224.npy')
output_path = '../car_market'
path = '../dataset'
to_delete = []
counter = 0


def read_img(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def transform(image):
    image = cv2.resize(image, (299, 224))
    image = cv2.remap(image, UNDIST_COORDS, None, cv2.INTER_LINEAR)
    image = image[0:224, 37:261]
    return image


def save_img(image, path):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_dir(path):
    global counter
    is_test = np.random.rand() < 0.1

    prefix = 'test' if is_test else 'train'
    prefix = os.path.join(output_path, prefix)
    prefix_q = os.path.join(output_path, 'query')
    prefix_mq = os.path.join(output_path, 'mquery')
    files = os.listdir(path)
    np.random.shuffle(files)

    for num, frame in enumerate(files):
        image = transform(read_img(os.path.join(path, frame)))
        image_name = '%06d_%03d.jpg' % (counter, num)
        save_img(image, os.path.join(prefix, image_name))

        if is_test:
            if np.random.rand() < 0.3:
                save_img(image, os.path.join(prefix_mq, image_name))
            if np.random.rand() < 0.1:
                save_img(image, os.path.join(prefix_q, image_name))

    counter += 1


for root, dirs, files in os.walk(path):
    if not dirs and not files:
        to_delete.append(root)

for dir in to_delete:
    os.removedirs(dir)

os.system(f'rm -rf {output_path}')
os.makedirs(output_path)
for subdir in ['train', 'test', 'query', 'mquery']:
    os.makedirs(os.path.join(output_path, subdir))

for store in os.listdir(path):
    for car in tqdm(os.listdir(os.path.join(path, store))):
        save_dir(os.path.join(path, store, car))
