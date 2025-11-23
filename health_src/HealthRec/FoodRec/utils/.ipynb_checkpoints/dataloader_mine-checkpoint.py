import pickle

import torch
import torch.utils.data as data
import random
import numpy as np

#from FoodRec.utils.utils import get_neg_ingre
from utils.utils import get_neg_ingre


class TrainDataLoader(data.Dataset):
    def __init__(self, args_config, dataset, use_neg_list=False):
        super(TrainDataLoader, self).__init__()
        self.args_config = args_config
        self.dataset = dataset
        self.n_ingredients = dataset.num_ingredients
        self.max_len = 20
        self.masked_p = 0.2
        self._user_input, self._item_input_pos, self._ingre_input_pos, self._ingre_num_pos, self._image_input_pos = self.init_samples()
        #self.use_neg_list = use_neg_list
        #self.neg_list = self.init_neg_list()
        #if args_config['health_neg_sample']:
        #    with open(args_config['graph_data_path'] + 'health_sample_dict.pkl', 'rb') as f:
        #        self.neg_sample_set, self.health_0, self.health_1, self.health_2, self.health_3, self.health_4, self.health_5 = pickle.load(f)

    def __len__(self):
        return len(self._user_input)

    def init_samples(self):
        _user_input, _item_input_pos, _ingre_input_pos, _ingre_num_pos, _image_input_pos = [], [], [], [], []
        for (u, i) in self.dataset.trainMatrix.keys():
            _user_input.append(u)
            _item_input_pos.append(i)
            _ingre_input_pos.append(self.dataset.ingredientCodeDict[i])
            _ingre_num_pos.append(self.dataset.ingredientNum[i])
            _image_input_pos.append(self.dataset.embImage[i])
        return _user_input, _item_input_pos, _ingre_input_pos, _ingre_num_pos, _image_input_pos

    #def init_neg_list(self):
    #    neg_list = []
    #    for user in self._user_input:
    #        pos_items = self.dataset.trainList[user]
    #        pos_validTest = self.dataset.validTestRatings[user]
    #        neg_item = self.get_random_neg(pos_items, pos_validTest)
    #        neg_list.append(neg_item)
    #    neg_lists = random.sample(neg_list, len(neg_list))    ->  shuffle을 왜 함?
    #    print("neg_list")
    #    print(neg_list[:10])
    #    print(neg_lists[:10])
    #    return neg_lists

    def __getitem__(self, index):
        # users,
        # pos_items, pos_image, pos_hl, pos_cate,
        # neg_items, neg_image, neg_hl, neg_cate,

        out_dict = {}
        u_id = self._user_input[index]
        out_dict['u_id'] = u_id
        pos_i_id = self._item_input_pos[index]
        out_dict['pos_i_id'] = pos_i_id
        out_dict['pos_ingre_code'] = self._ingre_input_pos[index]
        out_dict['pos_ingre_num'] = self._ingre_num_pos[index]
        out_dict['pos_img'] = self._image_input_pos[index]
        # out_dict['pos_ingre_emb'] = self.dataset.ingre_emb[pos_i_id]
        if self.args_config['SCHGN_ssl']:
            out_dict['masked_ingre_seq'], out_dict['pos_ingre_seq'], out_dict['neg_ingre_seq'] = self.ssl_task(
                out_dict['pos_ingre_code'], out_dict['pos_ingre_num'])
        #if self.use_neg_list:
        #   neg_i_id = self.neg_list[index]
        #else:
        pos_items = self.dataset.trainList[u_id]
        pos_validTest = self.dataset.validTestRatings[u_id]
        neg_i_id = self.get_random_neg(pos_items, pos_validTest)
        
        out_dict['neg_i_id'] = neg_i_id
        out_dict['neg_ingre_code'] = self.dataset.ingredientCodeDict[neg_i_id]
        out_dict['neg_ingre_num'] = self.dataset.ingredientNum[neg_i_id]
        out_dict['neg_img'] = self.dataset.embImage[neg_i_id]
        # out_dict['neg_ingre_emb'] = self.dataset.ingre_emb[neg_i_id]
        if self.args_config['use_cal_level']:
            out_dict['pos_cl'] = self.dataset.cal_level[pos_i_id]
            out_dict['neg_cl'] = self.dataset.cal_level[neg_i_id]
        if self.args_config['use_health_level']:
            out_dict['pos_hl'] = self.dataset.health_level[pos_i_id]
            out_dict['neg_hl'] = self.dataset.health_level[neg_i_id]
        if self.args_config['use_health_level_multi_hot']:
            out_dict['pos_hl_mh'] = torch.tensor(self.dataset.health_level_multi_hot[pos_i_id], dtype=torch.float)
            out_dict['neg_hl_mh'] = torch.tensor(self.dataset.health_level_multi_hot[neg_i_id], dtype=torch.float)
        #if self.args_config['health_neg_sample']:
        #    pos_items = self.dataset.trainList[u_id]
        #    pos_validTest = self.dataset.validTestRatings[u_id]

        #    while True:
        #        if u_id in self.neg_sample_set:
        #            if self.dataset.health_level[pos_i_id] == 0:
        #                health_neg = random.choice(self.health_0)
        #            elif self.dataset.health_level[pos_i_id] == 1:
        #                health_neg = random.choice(self.health_1)
        #            elif self.dataset.health_level[pos_i_id] == 2:
        #                health_neg = random.choice(self.health_2)
        #            elif self.dataset.health_level[pos_i_id] == 3:
        ##                health_neg = random.choice(self.health_3)
         #           elif self.dataset.health_level[pos_i_id] == 4:
        #                health_neg = random.choice(self.health_4)
        #            else:
        #                health_neg = random.choice(self.health_5)
        #        else:
        #            health_neg = random.choice(self.dataset.train_item_list)
        #        if health_neg not in pos_items and health_neg not in pos_validTest:
        #            break
        #    out_dict['health_neg'] = health_neg
        #    out_dict['health_neg_ingre_code'] = self.dataset.ingredientCodeDict[health_neg]
        #    out_dict['health_neg_ingre_num'] = self.dataset.ingredientNum[health_neg]
        #    out_dict['health_neg_img'] = self.dataset.embImage[health_neg]
        #    out_dict['health_neg_cl'] = self.dataset.cal_level[health_neg]
        #    out_dict['health_neg_hl'] = self.dataset.health_level[health_neg]
        return out_dict

    def ssl_task(self, ingre_seq, ingre_num):
        masked_ingre_seq = []
        neg_ingre = []
        pos_ingre = []
        ingre_set = set(ingre_seq[:ingre_num])
        for idx, ingre in enumerate(ingre_seq):
            if idx < ingre_num:
                pos_ingre.append(ingre)
                prob = random.random()
                if prob < self.masked_p:
                    masked_ingre_seq.append(self.n_ingredients + 1)
                    neg_ingre.append(get_neg_ingre(ingre_set, self.n_ingredients))
                else:
                    masked_ingre_seq.append(ingre)
                    neg_ingre.append(ingre)
            else:
                pos_ingre.append(ingre)
                masked_ingre_seq.append(ingre)
                neg_ingre.append(ingre)

        assert len(masked_ingre_seq) == self.max_len
        assert len(pos_ingre) == self.max_len
        assert len(neg_ingre) == self.max_len

        return torch.tensor(masked_ingre_seq, dtype=torch.long), \
            torch.tensor(pos_ingre, dtype=torch.long), \
            torch.tensor(neg_ingre, dtype=torch.long)

    def get_random_neg(self, train_pos, validTest_pos):
        while True:
            neg_i_id = np.random.randint(self.dataset.num_items)
            # neg_i_id = random.choice(self.dataset.train_item_list)
            if neg_i_id not in train_pos and neg_i_id not in validTest_pos:
                break
        return neg_i_id


class EvalDataLoader(data.Dataset):
    def __init__(self, args_config, dataset, phase='val', full_sort=False):
        super(EvalDataLoader, self).__init__()
        self.args_config = args_config
        self.full_sort = full_sort    # True
        self.dataset = dataset
        
        if not full_sort:
            self._user_input, self._item_input_pos, self._ingre_input_pos, self._ingre_num_pos, self._image_input_pos = (
                self.init_eval(phase))
        if phase == 'val':
            self.user_ids = self.dataset.valid_users
        else:
            self.user_ids = list(range(self.dataset.n_users))

    def __len__(self):
        if self.full_sort:
            return len(self.user_ids)
        else:
            return len(self._user_input)

    def init_eval(self, phase):
        _ingre_input_pos, _ingre_num_pos, _image_input_pos = [], [], []
        if phase == 'val':
            _user_input = list(self.dataset.valid_data[:, 0])
            #print(f"valid_data : {self.dataset.valid_data[:, 1][:10]}")
            
            #_item_input_pos = list(self.dataset.valid_data[:, 1] - self.dataset.n_users)
            _item_input_pos = list(self.dataset.valid_data[:, 1])
            #print(f"item_input_pos : {_item_input_pos[:10]}")
        else:
            _user_input = list(self.dataset.test_data[:, 0])
            #print(f"test_data : {self.dataset.test_data[:, 1][:10]}")
            
            #_item_input_pos = list(self.dataset.test_data[:, 1] - self.dataset.n_users)
            _item_input_pos = list(self.dataset.test_data[:, 1])
            #print(f"item_input_pos : {_item_input_pos[:10]}")
            
        assert len(_user_input) == len(_item_input_pos)
        for i in _item_input_pos:
            _ingre_input_pos.append(self.dataset.ingredientCodeDict[i])
            _ingre_num_pos.append(self.dataset.ingredientNum[i])
            _image_input_pos.append(self.dataset.embImage[i])
        return _user_input, _item_input_pos, _ingre_input_pos, _ingre_num_pos, _image_input_pos

    def __getitem__(self, index):
        # users,
        # pos_items, pos_image, pos_hl, pos_cate,
        # neg_items, neg_image, neg_hl, neg_cate,
        out_dict = {}
        if not self.full_sort:
            u_id = self._user_input[index]
            out_dict['u_id'] = u_id
            pos_i_id = self._item_input_pos[index]
            out_dict['pos_i_id'] = pos_i_id
            out_dict['pos_ingre_code'] = self._ingre_input_pos[index]
            out_dict['pos_ingre_num'] = self._ingre_num_pos[index]
            out_dict['pos_img'] = self._image_input_pos[index]

            #neg_items = self.dataset.testNegatives[u_id]
            #ingre_code_list, ingre_num_list, img_list = [], [], []
            #for i in neg_items:
            #    ingre_code_list.append(self.dataset.ingredientCodeDict[i])
            #    ingre_num_list.append(self.dataset.ingredientNum[i])
            #    img_list.append(self.dataset.embImage[i])
            #out_dict['neg_ingre_code'], out_dict['neg_ingre_num'], out_dict['neg_img'] = \
            #    torch.tensor(np.array(ingre_code_list), dtype=torch.long), torch.tensor(np.array(ingre_num_list), dtype=
            #    torch.long), torch.tensor(np.array(img_list), dtype=
            #    torch.float32)
            #out_dict['neg_i_id'] = torch.tensor(neg_items, dtype=torch.long)

            if self.args_config['use_cal_level']:
                out_dict['pos_cl'] = self.dataset.cal_level[pos_i_id]
                
                #cal_level_list = []
                #for i in neg_items:
                #    cal_level_list.append(self.dataset.cal_level[i])
                #out_dict['neg_cl'] = torch.tensor(np.array(cal_level_list), dtype=torch.long)

        else:
            u_id = self.user_ids[index]
            out_dict['u_id'] = u_id
        return out_dict