from typing import Optional
import dhash
from PIL import Image
import os
import numpy as np
from threading import Lock
from twinspect.metrics.hamming import HammingHero
import imagehash

# Create a Lock object
lock = Lock()

from functools import cache
from joblib import Memory

memory = Memory(".cache/pch/")


def image_dhash(fp) -> Optional[str]:
    image = Image.open(fp)
    row, col = dhash.dhash_row_col(image)
    return dhash.format_hex(row, col)[:16]


@cache
def get_clip_model():
    from transformers import CLIPProcessor, CLIPModel
    model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return clip_model, clip_processor


@memory.cache
def image_clip_embed(fp, greyscale=True) -> Optional[str]:
    import torch
    clip_model, clip_processor = get_clip_model()
    image = Image.open(fp)
    if greyscale:
        image = image.convert("L")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs


@memory.cache
def get_clip_embeds_df(n=1000):
    from tqdm import tqdm
    base_dir = "data/mirflickr_mfnd/"
    image_paths = []
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image_paths.append(image_path)
    np.random.shuffle(image_paths)
    clip_embeds = []
    image_paths_sample = image_paths[:n]
    for image_path in tqdm(image_paths_sample):
        clip_embeds.append(image_clip_embed(image_path)[0])
    return np.asarray(clip_embeds)


@memory.cache
def get_pca_model():
    from sklearn.decomposition import PCA
    pca_model = PCA(n_components=128)
    clip_embeds_df = get_clip_embeds_df()
    pca_embds_df = pca_model.fit_transform(clip_embeds_df)
    pca_mean_vec = pca_embds_df.mean(axis=0)
    pca_var_vec = np.var(pca_embds_df, axis=0)
    return pca_model, pca_mean_vec, pca_var_vec


@cache
def get_norm_distr_buckets(bucket_count):
    from scipy.stats import norm
    bucket_edges = []
    for i in range(1, bucket_count):
        quantile = i / bucket_count
        x_value = norm.ppf(quantile)
        bucket_edges.append(x_value)
    bucket_edges = [-np.inf] + bucket_edges + [np.inf]
    return bucket_edges


def _hex_to_counter(value):
    if value == 0:
        return value
    return (1 << value) - 1


def _encode_pch(clip_embed, size=32, bucket_count=4, grey_code=False, counter_bin=True):
    assert bucket_count in [4, 16, 256]

    from bisect import bisect_left
    pca_model, pca_mean_vec, pca_var_vec = get_pca_model()
    pca_embed = pca_model.transform(clip_embed)[0]
    vec = (pca_embed - pca_mean_vec) / pca_var_vec
    buckets = get_norm_distr_buckets(bucket_count)
    accum_shift = None
    if bucket_count == 4:
        accum_shift = 2
    if bucket_count <= 16:
        format_str = '{:01x}'
    else:
        format_str = '{:02x}'
    pch = []
    last_bucket = 0
    f = lambda x: x
    if grey_code:
        f = lambda x: x ^ (x >> 1)
    elif counter_bin:
        f = _hex_to_counter
        format_str = '{:01x}'
        accum_shift = None
    for i in range(size):
        bucket = f(min(bucket_count - 1, max(0, bisect_left(buckets, vec[i]) - 1)))
        if accum_shift is not None:
            if i & 1:
                last_bucket = (last_bucket << accum_shift) + bucket
                pch.append(format_str.format(last_bucket))
            else:
                last_bucket = bucket
        else:
            pch.append(format_str.format(bucket))
    return ''.join(pch)


def image_clip_pch_binary(fp, size) -> Optional[str]:
    with lock:
        clip_embed = image_clip_embed(fp)
        pca_model, pca_mean_vec, pca_var_vec = get_pca_model()
        pca_embed = pca_model.transform(clip_embed)[0]
        vec = (pca_embed - pca_mean_vec) / pca_var_vec
        return hex(int(''.join([str(int(i > 0)) for i in vec[:size]]), 2))[2:].rjust(size // 4, '0')


def image_clip_pch_binary_16(fp) -> Optional[str]:
    return image_clip_pch_binary(fp, 64)


def image_clip_pch(fp, size=32, bucket_count=4, counter_bin=True) -> Optional[str]:
    with lock:
        clip_embed = image_clip_embed(fp)
        return _encode_pch(clip_embed, size=size, bucket_count=bucket_count,
                           counter_bin=counter_bin)


def image_clip_pch_32_4(fp) -> Optional[str]:
    return image_clip_pch(fp, 32, 4)


def image_clip_pch_32_16(fp) -> Optional[str]:
    return image_clip_pch(fp, 32, 16)


def image_clip_pch_64_16(fp) -> Optional[str]:
    return image_clip_pch(fp, 64, 16)


def image_clip_pch_128_16(fp) -> Optional[str]:
    return image_clip_pch(fp, 128, 16)


def image_clip_pch_16_4(fp) -> Optional[str]:
    return image_clip_pch(fp, 16, 4)


def image_clip_pch_8_4(fp) -> Optional[str]:
    return image_clip_pch(fp, 8, 4)


def image_clip_pch_8_16(fp) -> Optional[str]:
    return image_clip_pch(fp, 8, 16)


def image_clip_pch_16_16(fp) -> Optional[str]:
    return image_clip_pch(fp, 16, 16)


def image_phash(fp) -> Optional[str]:
    return imagehash.phash(Image.open(fp))


def image_phash_simple(fp) -> Optional[str]:
    return imagehash.phash_simple(Image.open(fp))


def test(path='cluster_00000/585585.jpg'):
    fp = f"data/mirflickr_mfnd/{path}"
    print(image_clip_pch(fp))


def test2(path1='cluster_01603/767541.jpg', path2='cluster_01603/767662.jpg', size=32,
          bucket_count=4, counter_bin=True):
    from scipy.spatial.distance import hamming
    fp = f"data/mirflickr_mfnd/{path1}"
    h1 = image_clip_pch(fp, size, bucket_count, counter_bin)
    fp = f"data/mirflickr_mfnd/{path2}"
    h2 = image_clip_pch(fp, size, bucket_count, counter_bin)
    print(h1)
    print(h2)
    print(hamming(list(h1), list(h2)))


def test_clip_dist(path1='cluster_01603/767541.jpg', path2='cluster_01603/767662.jpg',
                   greyscale=False):
    from scipy.spatial.distance import cosine
    fp1 = f"data/mirflickr_mfnd/{path1}"
    fp2 = f"data/mirflickr_mfnd/{path2}"
    print(cosine(image_clip_embed(fp1, greyscale)[0], image_clip_embed(fp2, greyscale)[0]))


def test_clip_pca_dist(path1='cluster_01603/767541.jpg', path2='cluster_01603/767662.jpg',
                       greyscale=False):
    from scipy.spatial.distance import cosine
    pca_model, pca_mean_vec, pca_var_vec = get_pca_model()
    fp1 = f"data/mirflickr_mfnd/{path1}"
    fp2 = f"data/mirflickr_mfnd/{path2}"
    pca_embed1 = pca_model.transform(image_clip_embed(fp1, greyscale))[0]
    pca_embed2 = pca_model.transform(image_clip_embed(fp2, greyscale))[0]
    print(cosine(pca_embed1, pca_embed2))


def test_ground_truth(
    simprint_path='data/image_clip_pch-mirflickr_mfnd-b3ef5e210672947f-simprint.csv'):
    from twinspect.metrics.eff import compare_to_ground_truth
    compare_to_ground_truth(simprint_path)


def test_hnsw(simprint_path='data/image_clip_pch-mirflickr_mfnd-b3ef5e210672947f-simprint.csv'):
    from faiss import IndexBinaryHNSW
    hh = HammingHero(simprint_path)
    index: IndexBinaryHNSW = hh.index
    fp = "data/mirflickr_mfnd/cluster_00000/585585.jpg"
    code = image_clip_pch(fp)
    print(code)
    uint8_matrix = np.empty((1, len(code)), dtype=np.uint8)
    for j in range(0, len(code), 2):
        uint8_matrix[0, j // 2] = np.uint8(int(code[j: j + 2], 16))

    query_code = uint8_matrix
    print(query_code)
    threshold = 100
    distances, indices = index.search(query_code, threshold)
    print(distances, indices)
