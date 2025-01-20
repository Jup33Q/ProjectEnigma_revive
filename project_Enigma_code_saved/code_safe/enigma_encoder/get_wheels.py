import numpy as np
from .LETTER_DICT import Letter_Transformers
from random import shuffle
import os

class EmbeddingWheels:
    def __init__(self, letter_transform: Letter_Transformers=Letter_Transformers(),save_dir:str='./wheels/'):
        self.size=len(letter_transform)
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            print(f"目录{save_dir}已存在，忽略创建操作")
        a,b=self.get_shuffler()
        _path = os.path.join(save_dir, 'embeddingwheels.npz')
        np.savez(_path,shuffler=a,shuffle_inv=b)


    def get_shuffler(self):
        size=self.size
        l1=[_ for _ in range(size)]
        l2=l1.copy()
        shuffle(l2)
        shuffler=np.zeros([size,size],dtype=np.int8)
        shuffle_inv=np.copy(shuffler)
        shuffler[l1,l2]=1
        shuffle_inv[l2,l1]=1
        return shuffler,shuffle_inv

class Turner:
    def __init__(self, letter_transform: Letter_Transformers = Letter_Transformers(), save_dir: str = './wheels/'):
        self.size = len(letter_transform)
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            print(f"目录{save_dir}已存在，忽略创建操作")
        a, b = self.get_shuffler()
        _path = os.path.join(save_dir, 'turner.npz')
        np.savez(_path, shuffler=a, shuffle_inv=b)

    def get_shuffler(self):
        size = self.size
        l1 = [_ for _ in range(size)]
        l2 = l1.copy()
        shuffle(l2)
        shuffler = np.zeros([size, size], dtype=np.int8)
        shuffle_inv = np.copy(shuffler)
        shuffler[l1, l2] = 1
        shuffle_inv[l2, l1] = 1
        return shuffler, shuffle_inv

class TurnerWheels:
    def __init__(self,letter_transform: Letter_Transformers = Letter_Transformers(),save_dir: str = './wheels/'):
        size = len(letter_transform)
        wheel=np.eye(size,size,dtype=np.int8)
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            print(f"目录{save_dir}已存在，忽略创建操作")
        _path = os.path.join(save_dir, 'turnerwheels.npz')
        np.savez(_path,shuffler=wheel,shuffle_inv=wheel)

def get_all_wheels(letter_transform: Letter_Transformers=Letter_Transformers(),save_dir:str='./wheels/'):
    EmbeddingWheels(letter_transform,save_dir=save_dir)
    TurnerWheels(letter_transform,save_dir=save_dir)
    Turner(letter_transform,save_dir=save_dir)
if __name__ == '__main__':
    EmbeddingWheels()
    turner=Turner()
    TurnerWheels()



