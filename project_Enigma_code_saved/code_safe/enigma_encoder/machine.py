from .LETTER_DICT import Letter_Transformers
import numpy as np
import os
import struct


class EnigmaMachine:
    def __init__(self,letter_transformer:Letter_Transformers=Letter_Transformers(),wheels_dir:str='./wheels'):
        self.former = letter_transformer
        self.emb, self.emb_inv=self.get_wheels(wheels_dir,'embeddingwheels.npz')
        self.turner,self.turner_inv=self.get_wheels(wheels_dir,'turner.npz')
        self.wheel, self.wheel_inv = self.get_wheels(wheels_dir, 'turnerwheels.npz')
        self.diff=0 # +1 for each letter encoded, -1 for each letter decoded
        if os.path.exists(f"{wheels_dir}/diff.bin"):
            # print('---------------')
            self.load_diff(wheels_dir)


    @staticmethod
    def get_wheels(wheels_dir,fname):
        _path=f'{wheels_dir}/{fname}'
        print(_path)
        _loader=np.load(_path)
        return _loader['shuffler'],_loader['shuffle_inv']

    def encode_single_letter(self,letter):
        letter_ohe = self.former.batch_encode_letter2onehot(letter)
        # print(letter_ohe.shape)
        _ids = self.wheel @ self.emb @ letter_ohe
        #
        self.wheel = self.wheel @ self.turner
        self.diff += 1
        return self.former.batch_decode_onehot(_ids)


    def batch_encode(self,sentence):
        sentence_ohe=self.former.batch_encode_letter2onehot(sentence)
        encrypted_ohe = np.copy(sentence_ohe)
        for i in range(sentence_ohe.shape[-1]):
            _ids = encrypted_ohe[:, i]
            _ids = self.wheel @ self.emb @ _ids
            encrypted_ohe[:, i] = _ids
            self.wheel = self.wheel @ self.turner
            self.diff+=1
        return self.former.batch_decode_onehot(encrypted_ohe)

    def encode_stream(self,sentence):
        for letter in sentence:
            yield self.encode_single_letter(letter)


    def batch_decode(self,sentence):
        encrypted_ohe=self.former.batch_encode_letter2onehot(sentence)
        decrypted_ohe = np.copy(encrypted_ohe)
        for i in range(encrypted_ohe.shape[-1]):
            _ids = decrypted_ohe[:, i]
            _ids = self.emb_inv @ self.wheel_inv @ _ids
            decrypted_ohe[:, i] = _ids
            self.wheel_inv = self.turner_inv @ self.wheel_inv
            self.diff-=1
        return self.former.batch_decode_onehot(decrypted_ohe)


    def decode_single_letter(self,letter):
        _ids = self.former.batch_encode_letter2onehot(letter)
        # print(letter_ohe.shape)
        _ids = self.emb_inv @ self.wheel_inv @ _ids
        #
        self.wheel_inv = self.turner_inv @ self.wheel_inv
        self.diff-=1
        return self.former.batch_decode_onehot(_ids)

    def decode_stream(self,sentence):
        for letter in sentence:
            yield self.decode_single_letter(letter)

    def save_wheels(self,save_dir='./my_new_wheels'):
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            print(f"目录{save_dir}已存在，忽略创建操作")

        _path = os.path.join(save_dir, 'embeddingwheels.npz')
        np.savez(_path, shuffler=self.emb, shuffle_inv=self.emb_inv)
        _path = os.path.join(save_dir, 'turner.npz')
        np.savez(_path, shuffler=self.turner, shuffle_inv=self.turner_inv)
        _path = os.path.join(save_dir, 'turnerwheels.npz')
        np.savez(_path, shuffler=self.wheel, shuffle_inv=self.wheel_inv)
        _path = os.path.join(save_dir, 'diff.bin')
        with open(_path, 'wb') as f:
            f.write(struct.pack('i', self.diff))

    def load_diff(self,wheels_dir='./wheels'):
        with open(os.path.join(wheels_dir, 'diff.bin'), 'rb') as f:
            loaded_num, = struct.unpack('i', f.read())
        self.diff = loaded_num

    def show_encoded_letter_without_turn(self,letter:str):
        letter_ohe = self.former.batch_encode_letter2onehot(letter)
        rt = dict()
        _ids1 =  self.emb @ letter_ohe
        _ids2= self.wheel @ _ids1
        rt['letter']=[letter]
        rt['after_embedding']=[self.former.batch_decode_onehot(_ids1)]
        rt['after_turner']=[self.former.batch_decode_onehot(_ids2)]

        return rt
    def show_decoded_letter_without_turn(self,letter:str):
        encrypted_ohe = self.former.batch_encode_letter2onehot(letter)
        rt = dict()
        _ids1 =  self.wheel_inv @ encrypted_ohe
        _ids2= self.emb_inv @ _ids1
        rt['letter']=[letter]
        rt['before_turner']=[self.former.batch_decode_onehot(_ids1)]
        rt['before_embedding']=[self.former.batch_decode_onehot(_ids2)]

        return rt




if __name__ == '__main__':
    machine=EnigmaMachine()
    print(machine.wheel.shape)
    # print(machine.encode_single_letter('a'))
    # machine.decode_single_letter('a')
    print(machine.show_encoded_letter_without_turn('a'))
    print(machine.show_encoded_letter_without_turn('a'))

    sentence='hello world'
    # gen=machine.batch_encode_stream(sentence)
    # s1=''
    # for i in gen:
    #     print(i)
    #     s1+=i


    s1=machine.batch_encode(sentence)
    print('|'+s1+'|')
    print(len(s1))

    print(machine.batch_decode(s1))

    machine.save_wheels()

    enigma2=EnigmaMachine(wheels_dir='./my_new_wheels')
    print(enigma2.diff)





