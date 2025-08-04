import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from predictor_models import *
from torch.optim import lr_scheduler
from modules import *

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--hidden-dim', type=int, default=256)
parser.add_argument('--save-folder', type=str, default='trained_predictor/')
parser.add_argument('--suffix', type=str, default='_springs6')
parser.add_argument('--num-atoms', type=int, default=6)
parser.add_argument('--num-hidden', type=int, default=1)
parser.add_argument('--timesteps', type=int, default=49)
parser.add_argument('--dims', type=int, default=4)
parser.add_argument('--model_type', type=str, default='ssb')
parser.add_argument('--use_gt_struc', action='store_true', default=True)
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='weight_decay.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout of set transformer.')
parser.add_argument('--pre_models_path', type=str, default='pre_models',
                    help='Whether test with dynamically re-computed graph.')
if __name__ == '__main__':
    ss=1.
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.num_visible = args.num_atoms - args.num_hidden
    os.makedirs(args.save_folder, exist_ok=True)
    if args.model_type == 'mlp':
        input_dim = args.num_visible * args.timesteps * args.dims
    elif args.model_type == 'sla' or args.model_type == 'set' or args.model_type == 'ssb':
        input_dim = args.timesteps * args.dims
    else:
        input_dim = args.dims

    if args.model_type == 'mlp':
        predictor = HiddenStatePredictorMLP(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.dims,
            num_timesteps=args.timesteps,
            num_hidden_nodes=args.num_hidden,
        )
    elif args.model_type == 'tra':
        predictor = HiddenStatePredictorTransformer(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.dims,
            num_timesteps=args.timesteps,
            num_hidden_nodes=args.num_hidden,
        )
    elif args.model_type == 'gnn':
        predictor = GNNHiddenStatePredictor(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.dims,
            num_timesteps=args.timesteps,
            num_hidden_nodes=args.num_hidden,
            num_visible_nodes=args.num_visible
        )
    elif args.model_type == 'gat':
        predictor = GATHiddenStatePredictor(
            input_dim=input_dim, hidden_dim=args.hidden_dim ,output_dim=args.dims,
            num_timesteps=args.timesteps,
            num_visible_nodes=args.num_visible,
            num_hidden_nodes=args.num_hidden
        )
    elif args.model_type == 'rnn':
        predictor = GNN_RNN_HiddenStatePredictor(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.dims,
            num_timesteps=args.timesteps,
            num_hidden_nodes=args.num_hidden,
            num_visible_nodes=args.num_visible
        )
    elif args.model_type == 'sla':
        predictor = SlotAttention(
            dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_slots=args.num_hidden,
        )
    elif args.model_type == 'set':
        predictor = SetTransformerHSP(
            input_dim=input_dim,
            num_timesteps=args.timesteps,
            num_slots=args.num_hidden,
            output_dim=args.dims,
            num_inds=32,
            dim_hidden=args.hidden_dim,
            num_heads=4,
            ln=True,
            dropout=args.dropout,
        )
    elif args.model_type == 'ssb':
        predictor = SetTransformerHSP_sb(
            input_dim=input_dim,
            num_timesteps=args.timesteps,
            num_slots=args.num_hidden,
            output_dim=args.dims,
            num_inds=32,
            dim_hidden=args.hidden_dim,
            num_heads=4,
            ln=True,
            dropout=args.dropout,
        )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # Load dataset

    train_loader, valid_loader,test_loader , _, _, _, _ = load_data(
        args.batch_size, args.suffix
    )
    if args.cuda:
        predictor.cuda()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)
    loss_fn = nn.MSELoss()
    # Training loop
    best_loss = np.inf

    # 遍历整个训练集，计算初始结构并缓存
    pre_hsp = SetTransformerHSP(
        input_dim=196,
        num_timesteps=args.timesteps,
        num_slots=args.num_hidden,
        output_dim=args.dims,
        num_inds=32,
        dim_hidden=args.hidden_dim,
        num_heads=4,
        ln=True,
        dropout=args.dropout,
    )
    pre_hsp_ckpt = os.path.join(args.pre_models_path, 'pre_hsp_n6.pth')
    pre_hsp.load_state_dict(torch.load(pre_hsp_ckpt))
    pre_hsp.cuda()
    pre_hsp.eval()
    pre_nri_enc = MLPEncoder(args.timesteps * args.dims, args.hidden_dim,
                             2,
                            0., True)
    pre_nri_enc_ckpt = os.path.join(args.pre_models_path, 'pre_nri_enc_n6.pt')
    pre_nri_enc.load_state_dict(torch.load(pre_nri_enc_ckpt))
    pre_nri_enc.cuda()
    pre_nri_enc.eval()
    off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec)
    rel_send = torch.FloatTensor(rel_send)
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    cached_adj_list = []
    ini_acc_list=[]
    ini_hsp_list=[]
    for batch in train_loader:
        data, relations = batch  # data shape: [B, N, T, D]
        data = data.cuda()
        relations = relations.cuda()
        visible_data = data[:, :args.num_visible]
        true_hidden = data[:, args.num_visible:]
        with torch.no_grad():
            pred_hidden = pre_hsp(visible_data)  # [B, H, T, D]
            pred_hidden_aligned, perm = hungarian_match(pred_hidden, true_hidden)
            data_all = torch.cat([visible_data, pred_hidden_aligned], dim=1)  # [B, N, T, D]
            logits = ss*pre_nri_enc(data_all, rel_rec, rel_send)  # [B, E, num_edge_types]
            pred_relations = torch.argmax(logits, dim=-1)  # [B, E]
            adj = unflatten_adj_matrix(pred_relations, num_nodes=args.num_atoms)  # [B, N, N]
            # adj_aligned=align_structure(adj, perm,args.num_visible)
            ini_acc = edge_accuracy(logits, relations)
            ini_acc_list.append(ini_acc)
            ini_loss = loss_fn(pred_hidden_aligned, true_hidden)
            ini_hsp_list.append(ini_loss.item())
        for i in range(data.shape[0]):
            cached_adj_list.append(adj[i].cpu())  # 存入结构列表，放在 CPU 上节省显存

    print(np.mean(ini_acc_list))
    print(np.mean(ini_hsp_list))

    cached_adj_list_val = []
    ini_acc_list_val = []
    ini_hsp_list_val = []
    # 遍历整个验证集，计算初始结构并缓存
    for batch in valid_loader:


        data, relations = batch  # data shape: [B, N, T, D]
        data = data.cuda()

        data = data[:, :, :49, :]
        relations = relations.cuda()
        visible_data = data[:, :args.num_visible]
        true_hidden = data[:, args.num_visible:]
        with torch.no_grad():
            pred_hidden = pre_hsp(visible_data)  # [B, H, T, D]
            pred_hidden_aligned, perm = hungarian_match(pred_hidden, true_hidden)
            data_all = torch.cat([visible_data, pred_hidden_aligned], dim=1)  # [B, N, T, D]
            logits = ss*pre_nri_enc(data_all, rel_rec, rel_send)  # [B, E, num_edge_types]
            pred_relations = torch.argmax(logits, dim=-1)  # [B, E]
            adj = unflatten_adj_matrix(pred_relations, num_nodes=args.num_atoms)  # [B, N, N]
            # adj_aligned = align_structure(adj, perm, args.num_visible)
            ini_acc_val = edge_accuracy(logits, relations)
            ini_acc_list_val.append(ini_acc_val)
            ini_loss_val = loss_fn(pred_hidden_aligned, true_hidden)
            ini_hsp_list_val.append(ini_loss_val.item())
        for i in range(data.shape[0]):
            cached_adj_list_val.append(adj[i].cpu())  # 存入结构列表，放在 CPU 上节省显存
    print(np.mean(ini_acc_list_val))
    print(np.mean(ini_hsp_list_val))

    for epoch in range(1, args.epochs + 1):
        predictor.train()
        epoch_loss = []
        update_acc_list=[]
        t_start = time.time()
        start_update_epoch = 40
        update_interval = 10
        update_structure = (epoch >= start_update_epoch) and (epoch % update_interval == 0)
        # if epoch == 20:
        #     update_structure=True
        for batch_idx, (data, relations) in enumerate(train_loader):
            if args.cuda:
                data = data.cuda()
                relations=relations.cuda()
            B = data.shape[0]
            sample_indices = batch_idx * args.batch_size + torch.arange(B)
            cached_adj_batch = [cached_adj_list[idx] for idx in sample_indices]
            cached_adj_batch = torch.stack(cached_adj_batch).to(data.device)
            visible_data = data[:, :args.num_visible, :, :]  # [B, V, T, D]
            true_hidden = data[:, args.num_visible:, :, :]  # [B, H, T, D]
            if update_structure:
                predictor.eval()
                with torch.no_grad():
                    update_pred_hidden = predictor(visible_data, cached_adj_batch)  # 用旧结构估计隐藏状态
                    update_pred_hidden_aligned, perm = hungarian_match(update_pred_hidden, true_hidden)
                    update_data_all = torch.cat([visible_data, update_pred_hidden_aligned], dim=1)
                    update_logits = ss*pre_nri_enc(update_data_all, rel_rec, rel_send)
                    update_pred_relations = torch.argmax(update_logits, dim=-1)
                    update_adj = unflatten_adj_matrix(update_pred_relations, num_nodes=args.num_atoms)
                    update_acc = edge_accuracy(update_logits, relations)
                # 覆盖结构缓存
                for i, idx in enumerate(sample_indices):
                    cached_adj_list[idx] = update_adj[i].cpu()
                predictor.train()
            # mask_tl, mask_br, mask_others=create_masks(args.num_atoms, args.num_visible)
            # adj = unflatten_adj_matrix(relations, num_nodes=args.num_atoms)
            # adj[mask_br.unsqueeze_(0).expand(adj.shape[0], -1, -1)]=1
            # adj[mask_others.unsqueeze_(0).expand(adj.shape[0], -1, -1)]=1
            # adj=block_fill(adj,args.num_visible)

            if update_structure:
                update_acc_list.append(update_acc)
            # perm = torch.randperm(args.num_hidden)
            # true_hidden = true_hidden[:, perm, :, :]
            # print(visible_data.shape)
            if args.use_gt_struc and (args.model_type == 'gnn' or args.model_type =='gat' or args.model_type =='rnn'or args.model_type =='ssb'):
                pred_hidden = predictor(visible_data,cached_adj_batch)
            else:
                pred_hidden = predictor(visible_data)
            # pred_hidden_aligned,_ = hungarian_match(pred_hidden, true_hidden)
            # loss = loss_fn(pred_hidden_aligned, true_hidden)
            loss = loss_fn(pred_hidden, true_hidden)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
        scheduler.step()
        print(f"Epoch {epoch:03d} | Train Loss: {np.mean(epoch_loss):.6f} | Time: {time.time() - t_start:.2f}s")
        if update_structure:
            print(np.mean(np.array(update_acc_list)))
        # Optional: validation
        predictor.eval()
        val_loss = []
        val_loss_unaligned=[]

        with torch.no_grad():
            for batch_idx, (data, relations) in enumerate(valid_loader):
                if args.cuda:
                    data = data.cuda()
                    relations = relations.cuda()
                adj = unflatten_adj_matrix(relations, num_nodes=args.num_atoms)
                data = data[:, :, :49, :]
                visible_data = data[:, :args.num_visible, :, :]
                true_hidden = data[:, args.num_visible:, :, :]
                # perm = torch.randperm(args.num_hidden)
                # true_hidden = true_hidden[:, perm, :, :]
                B = data.shape[0]
                sample_indices_val = batch_idx * args.batch_size + torch.arange(B)
                cached_adj_batch_val = [cached_adj_list_val[idx] for idx in sample_indices_val]
                cached_adj_batch_val = torch.stack(cached_adj_batch_val).to(data.device)
                if update_structure:
                    with torch.no_grad():
                        update_pred_hidden = predictor(visible_data, cached_adj_batch_val)  # 用旧结构估计隐藏状态
                        update_pred_hidden_aligned, perm = hungarian_match(update_pred_hidden, true_hidden)
                        update_data_all = torch.cat([visible_data, update_pred_hidden_aligned], dim=1)
                        update_logits =ss* pre_nri_enc(update_data_all, rel_rec, rel_send)
                        update_pred_relations = torch.argmax(update_logits, dim=-1)
                        update_adj = unflatten_adj_matrix(update_pred_relations, num_nodes=args.num_atoms)
                        update_acc = edge_accuracy(update_logits, relations)
                    for i, idx in enumerate(sample_indices_val):
                        cached_adj_list_val[idx] = update_adj[i].cpu()
                if args.use_gt_struc and (args.model_type == 'gnn' or args.model_type =='gat' or args.model_type =='rnn'or args.model_type =='ssb'):
                    pred_hidden = predictor(visible_data, cached_adj_batch_val)
                else:
                    pred_hidden = predictor(visible_data)
                pred_hidden_aligned ,_= hungarian_match(pred_hidden, true_hidden)
                val_loss.append(loss_fn(pred_hidden_aligned, true_hidden).item())
                val_loss_unaligned.append(loss_fn(pred_hidden, true_hidden).item())
                # val_loss.append(loss_fn(pred_hidden, true_hidden).item())
        avg_val_loss = np.mean(val_loss)
        avg_val_loss_unaligned = np.mean(val_loss_unaligned)
        print(f"Validation Loss: {avg_val_loss:.6f}")
        print(f"Validation Loss Unaligned: {avg_val_loss_unaligned:.6f}")
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(predictor.state_dict(), os.path.join(args.save_folder, args.model_type+str(args.num_atoms)+"predictor_best.pth"))
            print("Saved best model.")
    print("Training finished.")
