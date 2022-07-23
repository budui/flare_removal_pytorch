import multiprocessing as mp
import lmdb
import fire
from pathlib import Path
import pickle
from io import BytesIO
from tqdm import tqdm
from loguru import logger
from functools import partial


def pickle_buffer(obj):
    buffer = BytesIO()
    pickle.dump(obj, buffer)
    return buffer.getvalue()


def is_target_file(f):
    return Path(f).suffix == ".npy"


def load_file(p, root):
    return (
        p.relative_to(root).as_posix().encode("utf-8"),
        p.read_bytes(),
    )


def prepare(env, root, n_workers):
    files = []
    for p in root.glob("**/*"):
        if is_target_file(p):
            files.append(p)

    filenames = [p.relative_to(root).as_posix() for p in files]
    with env.begin(write=True) as txn:
        txn.put("filenames".encode("utf-8"), pickle_buffer(filenames))

    logger.info(f"will write {len(files)} files into lmdb dataset")

    load_file_fn = partial(load_file, root=root)

    with mp.Pool(n_workers) as pool:
        for key, data in tqdm(
            pool.imap_unordered(load_file_fn, files),
            total=len(files),
        ):
            with env.begin(write=True) as txn:
                txn.put(key, data)


def main(root, out, n_workers=8):
    root = Path(root)
    assert root.exists()
    with lmdb.open(out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, root, n_workers)


if __name__ == "__main__":
    fire.Fire(main)
