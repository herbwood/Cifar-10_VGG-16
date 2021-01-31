import pickle
import requests
import tarfile

def unpickle(file : str):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def download_dataset(url : str = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
             target_path : str = 'cifar-10-python.tar.gz'):

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())

    if target_path.endswith("tar.gz"):

        tar = tarfile.open(target_path, "r:gz")
        tar.extractall()
        tar.close()

if __name__ == "__main__":
    download_dataset()