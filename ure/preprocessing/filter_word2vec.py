import sys
from ure.vocabulary import Vocabulary
import ure.utils as utils
import numpy as np


if __name__ == "__main__":
    core_voca_path = sys.argv[1]
    word_embs_dir = sys.argv[2]
    keep_voca_path = sys.argv[3]

    print('load core voca from', core_voca_path)
    core_voca, _ = Vocabulary.load(core_voca_path, normalization=True, add_pad_unk=True, lower=True, digit_0=False)
    print('load core voca from', core_voca_path)
    print ('core voca size', core_voca.size())
    keep_voca, _ = Vocabulary.load(keep_voca_path, add_pad_unk=False, normalization=True, lower=True, digit_0=False)
    print('keep voca size', keep_voca.size())

    print('load full voca and embs')
    full_voca, full_embs = utils.load_voca_embs(
        '{}/all_dict.word'.format(word_embs_dir),
        '{}/all_word_embeddings.npy'.format(word_embs_dir))
    print('full voca size', full_voca.size())

    print ('add entity types')
    

    print('select word ids')
    selected = []
    for word in core_voca.id2word:
        word_id = full_voca.word2id.get(word, -1)
        if word_id >= 0:
            selected.append(word_id)

    need_generated = []
    for word in keep_voca.id2word:
        word_id = full_voca.word2id.get(word, -1)
        if word_id >= 0 and word_id not in selected:
            selected.append(word_id)
        else:
            need_generated.append(word)
            

    print('save...')
    selected_embs = full_embs[selected, :]
    generated_embeddings = np.random.normal(
        0, 1, size=(len(need_generated), full_embs.shape[1]))
    generated_embeddings = np.vstack((selected_embs, generated_embeddings))
    print('Pretrained embeddings {}\tTotal embeddings: {}'.format(selected_embs.shape, generated_embeddings.shape))

    np.save('{}/word_embeddings_etype_lower'.format(word_embs_dir), 
            generated_embeddings)

    with open('{}/dict.word.etype.lower'.format(word_embs_dir), 'w') as f:
        for i in selected:
            f.write(full_voca.id2word[i] + '\n')
        for w in need_generated:
            f.write(w + '\n')
