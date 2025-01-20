from typing import Annotated
import numpy as np


class Letter_Transformers:
    letters = r"qwertyuiopasdfghjklzxcvbnm,./;'[]\|+=_-~`1234567890!@#$%^&*(){}QWERTYUIOPASDFGHJKLZXCVBNM<>:? 你好"
    def __init__(self,letters=None):
        if letters is not None:
            self.letters = letters
        self.letter2num_dict=self.get_letter2num_dict()
        self.num2letter_dict=self.get_num2letter_dict()
    def get_zipper(self):
        letters= self.letters
        l=len(letters)
        letter_zippers=[(letters[i],i) for i in range(l)]
        return letter_zippers

    def get_letter2num_dict(self):
        rt=dict()
        for k,v in self.get_zipper():
            rt[k]=v
        return rt

    def get_num2letter_dict(self):
        rt=dict()
        for v,k in self.get_zipper():
            rt[k]=v
        return rt
    def __len__(self):
        return len(self.letters)

    def batch_decode_num2letter(self,arr:[list[int]|np.ndarray[int]]):
        rt_list=[self.num2letter_dict[num] for num in arr]
        return ''.join(rt_list)

    def batch_encode_letter2num(self,sentence:str)->list:
        rt_list=[self.letter2num_dict[c] for c in sentence]

        return rt_list

    def batch_encode_letter2onehot(self,sentence:str)->np.ndarray:
        size_ids=len(self)
        len_seq=len(sentence)
        rt=np.zeros((size_ids,len_seq),dtype=np.int8)

        cols=np.arange(len_seq)
        rows=[self.letter2num_dict[letter] for letter in sentence]
        rt[rows,cols]=1
        return rt

    def batch_decode_onehot(self,arr:[np.ndarray[int]]):
        t1 = np.where(arr == 1)
        d = dict()
        seq = arr.shape[-1]
        for i in range(seq):
            d[t1[1][i]] = t1[0][i]
        return ''.join([self.num2letter_dict[d[i]] for i in range(seq)])



if __name__ == '__main__':
    my_letter_transform=Letter_Transformers()

    # print(my_letter_transform.get_zipper())
    # out: [('q', 0), ('w', 1), ('e', 2), ('r', 3), ('t', 4), ('y', 5), ('u', 6), ('i', 7), ('o', 8), ('p', 9), ('a', 10), ...]

    # print(len(my_letter_transform))
    # out: 93
    # letter2num_dict=my_letter_transform.get_letter2num_dict()
    # print(letter2num_dict['c'])
    #out: 21

    # print(my_letter_transform.batch_decode_num2letter([0,1,2,3,4,5]))
    #out: qwert

    l1=my_letter_transform.batch_encode_letter2num(sentence='hello world')
    print(l1)
    # [15, 2, 18, 18, 8, 92, 1, 8, 3, 18, 12]
    sentence_decoded=my_letter_transform.batch_decode_num2letter(l1)
    print(sentence_decoded)
    # hello world

    arr=my_letter_transform.batch_encode_letter2onehot(sentence='hello world')
    # print(arr)

    print(my_letter_transform.batch_decode_onehot(arr))

    print(my_letter_transform.batch_encode_letter2onehot(sentence='h'))



