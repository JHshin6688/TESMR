import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization

class LGCN_CL_GD(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LGCN_CL_GD, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
        self.feat_embed_dim = config['feat_embed_dim']
        self.dropout_rate = config['dropout_rate']
        
        # hyperparameter for contrastive learning
        self.user_temperature = 0.05
        self.item_temperature = 0.05
        self.user_weight = 1e-04
        self.item_weight = 1e-04
        self.combine_strategy = 'attention'
        
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.embedding_dict, self.embedding_dict_pretrained, self.t_layers = self._init_model()
        self._init_combination_params()
        
        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat(dropout = False).to(self.device)
        self.norm_adj_matrix_dropout = None
        
        # parameters initialization
        #self.apply(xavier_uniform_initialization)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.latent_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.latent_dim)))
        })

        user_path = "../pretrained_data/user_embedding.pt"
        item_path = "../pretrained_data/item_embedding.pt"

        # dimension of pretrained embedding is 384
        user_embedding = nn.Embedding(self.n_users, 384)
        item_embedding = nn.Embedding(self.n_items, 384)
        
        user_embedding.load_state_dict(torch.load(user_path))
        item_embedding.load_state_dict(torch.load(item_path))

            
        embedding_dict_pretrained = nn.ParameterDict({
            'user_emb': user_embedding.weight,
            'item_emb': item_embedding.weight
        })

        layers = nn.ModuleDict({
            'user_trs_1' : nn.Linear(384, self.feat_embed_dim),
            'user_trs_2' : nn.Linear(self.latent_dim, self.feat_embed_dim),
            'item_trs_1' : nn.Linear(384, self.feat_embed_dim),
            'item_trs_2' : nn.Linear(self.latent_dim, self.feat_embed_dim)
        })
        
        return embedding_dict, embedding_dict_pretrained, layers

    def _init_combination_params(self):
        """Initialize parameters for different combination strategies"""
        if self.combine_strategy == 'weighted':
            self.user_combine_weight = nn.Parameter(torch.tensor(0.5))
            self.item_combine_weight = nn.Parameter(torch.tensor(0.5))
        elif self.combine_strategy == 'concat':
            self.user_combine_layer = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
            self.item_combine_layer = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
        elif self.combine_strategy == 'attention':
            self.user_attention = nn.MultiheadAttention(
                embed_dim=self.feat_embed_dim, num_heads=1, batch_first=True
            )
            self.item_attention = nn.MultiheadAttention(
                embed_dim=self.feat_embed_dim, num_heads=1, batch_first=True
            )

    def generate_dropout_matrix(self):
        """Apply new dropout to adj matrix for each epoch"""
        self.norm_adj_matrix_dropout = self.get_norm_adj_mat(dropout = True).to(self.device)
        
    def get_norm_adj_mat(self, dropout):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        interaction_matrix = self.interaction_matrix
        if dropout:
            adj = interaction_matrix.tocoo()
            keep_prob = 1.0 - self.dropout_rate
            n_edges = len(adj.data)
            keep_mask = np.random.rand(n_edges) < keep_prob

            kept_rows = adj.row[keep_mask]
            kept_cols = adj.col[keep_mask]
            kept_data = adj.data[keep_mask]

            interaction_matrix = sp.coo_matrix(
                (kept_data, (kept_rows, kept_cols)),
                shape = adj.shape
            )
            
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        for (i, j) in zip(inter_M.row, inter_M.col):
            A[i, j + self.n_users] = 1
        for (i, j) in zip(inter_M_t.row, inter_M_t.col):
            A[i + self.n_users, j] = 1
        
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.from_numpy(np.array([row, col])).long()
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, size = torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        return ego_embeddings

    def combine_embeddings(self, emb1, emb2, entity_type='user'):
        """Combine two embeddings using the specified strategy"""
        if self.combine_strategy == 'add':
            return torch.add(emb1, emb2)
            
        elif self.combine_strategy == 'weighted':
            if entity_type == 'user':
                alpha = torch.sigmoid(self.user_combine_weight)
            else:
                alpha = torch.sigmoid(self.item_combine_weight)
            return alpha * emb1 + (1 - alpha) * emb2
            
        elif self.combine_strategy == 'concat':
            concat_emb = torch.cat([emb1, emb2], dim=-1)
            if entity_type == 'user':
                return self.user_combine_layer(concat_emb)
            else:
                return self.item_combine_layer(concat_emb)
                
        elif self.combine_strategy == 'attention':
            # Stack embeddings for attention
            stacked = torch.stack([emb1, emb2], dim=1)  # [N, 2, D]
            if entity_type == 'user':
                attended, _ = self.user_attention(stacked, stacked, stacked)
            else:
                attended, _ = self.item_attention(stacked, stacked, stacked)
            # Take weighted average
            return torch.mean(attended, dim=1)
            
        else:
            raise ValueError(f"Unknown combine strategy: {self.combine_strategy}")

    def forward(self, training):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        if training:
            adj_matrix = self.norm_adj_matrix_dropout
        else:
            adj_matrix = self.norm_adj_matrix
             
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
            
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_int_raw = lightgcn_all_embeddings[:self.n_users, :]
        item_int_raw = lightgcn_all_embeddings[self.n_users:, :]


        user_pre_embeddings = self.t_layers['user_trs_1'](self.embedding_dict_pretrained['user_emb'])
        user_int_embeddings = self.t_layers['user_trs_2'](user_int_raw)
        
        item_pre_embeddings = self.t_layers['item_trs_1'](self.embedding_dict_pretrained['item_emb'])
        item_int_embeddings = self.t_layers['item_trs_2'](item_int_raw)

        user_final_embeddings = self.combine_embeddings(user_pre_embeddings, user_int_embeddings, 'user')
        item_final_embeddings = self.combine_embeddings(item_pre_embeddings, item_int_embeddings, 'item')
        
        return (user_final_embeddings, item_final_embeddings, 
                user_pre_embeddings, user_int_embeddings, 
                item_pre_embeddings, item_int_embeddings)

    def compute_infonce_loss(self, emb1, emb2, temperature, indices=None):
        """
        Compute InfoNCE loss between two embedding modalities
        
        Args:
            emb1: First modality embeddings [B, D]
            emb2: Second modality embeddings [B, D]  
            temperature: Temperature parameter
            indices: Indices for sampling (if None, use all)
        """
        if indices is not None:
            emb1 = emb1[indices]
            emb2 = emb2[indices]
        
        # Normalize embeddings
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        batch_size = emb1.size(0)
        all_sim = torch.matmul(emb1, emb2.T) / temperature  # [B, B]
        labels = torch.arange(batch_size, device = emb1.device)

        # Compute InfoNCE loss
        loss = F.cross_entropy(all_sim, labels)
        return loss

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]
         
        (user_final_embeddings, item_final_embeddings, 
         user_pre_embeddings, user_int_embeddings, 
         item_pre_embeddings, item_int_embeddings) = self.forward(training = True)
        
        u_embeddings = user_final_embeddings[user, :]
        posi_embeddings = item_final_embeddings[pos_item, :]
        negi_embeddings = item_final_embeddings[neg_item, :]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.embedding_dict['user_emb'][user, :]
        posi_ego_embeddings = self.embedding_dict['item_emb'][pos_item, :]
        negi_ego_embeddings = self.embedding_dict['item_emb'][neg_item, :]
        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)

        # InfoNCE Loss: User modalities alignment
        user_nce = self.compute_infonce_loss(
            user_pre_embeddings, user_int_embeddings, 
            self.user_temperature, indices=user
        )
    
        # InfoNCE Loss: Item modalities alignment  
        item_indices = torch.cat([pos_item, neg_item])  # Include both pos and neg items
        item_nce = self.compute_infonce_loss(
            item_pre_embeddings, item_int_embeddings,
            self.item_temperature, indices=item_indices
        )
        
        # Total loss
        loss = (mf_loss + 
                self.reg_weight * reg_loss + 
                self.user_weight * user_nce + 
                self.item_weight * item_nce)
        
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        (user_final_embeddings, item_final_embeddings, 
         _, _, _, _) = self.forward(training = False)
        
        u_embeddings = user_final_embeddings[user, :]

        # Dot product with all item embeddings
        scores = torch.matmul(u_embeddings, item_final_embeddings.transpose(0, 1))

        return scores