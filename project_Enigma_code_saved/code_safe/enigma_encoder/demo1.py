import numpy as np
from fontTools.misc.eexec import encrypt

from LETTER_DICT import Letter_Transformers

if __name__ == '__main__':
    emb_loader=np.load('./wheels/embeddingwheels.npz')
    _ = (emb := emb_loader['shuffler'])
    _ = (emb_inv := emb_loader['shuffle_inv'])
     #EYE
    turner_loader=np.load('./wheels/turner.npz')
    _ = (turner := turner_loader['shuffler'])
    _ = (turner_inv := turner_loader['shuffle_inv'])
    wheel_loader=np.load('./wheels/turnerwheels.npz')
    _ = (wheel := wheel_loader['shuffler'])
    _ = (wheel_inv := wheel_loader['shuffle_inv'])

    sentence='my name is Jupiter?)(*&^'
    former=Letter_Transformers()
    sentence_ohe=former.batch_encode_letter2onehot(sentence)
    encrypted_ohe=np.copy(sentence_ohe)
    for i in range(sentence_ohe.shape[-1]):
        _ids= encrypted_ohe[:,i]
        _ids=wheel@emb@_ids
        encrypted_ohe[:,i]=_ids
        wheel=wheel@turner

    # print(encrypted_ohe)
    print('|'+former.batch_decode_onehot(encrypted_ohe)+"|")

    decrypted_ohe=np.copy(encrypted_ohe)
    for i in range(sentence_ohe.shape[-1]):
        _ids= decrypted_ohe[:,i]
        _ids=emb_inv@wheel_inv@_ids
        decrypted_ohe[:,i]=_ids
        wheel_inv=wheel_inv@turner_inv
    # print(decrypted_ohe)
    print('|'+former.batch_decode_onehot(decrypted_ohe)+"|")