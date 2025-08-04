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


from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=250)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--hidden-dim', type=int, default=256)
parser.add_argument('--save-folder', type=str, default='trained_predictor/')
parser.add_argument('--suffix', type=str, default='_springs10')
parser.add_argument('--num-atoms', type=int, default=10)
parser.add_argument('--num-hidden', type=int, default=5)
parser.add_argument('--timesteps', type=int, default=49)
parser.add_argument('--dims', type=int, default=4)
parser.add_argument('--model_type', type=str, default='set')
parser.add_argument('--use_gt_struc', action='store_true', default=True)
parser.add_argument('--lr-decay', type=int, default=100,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--weight_decay', type=float, default=0.,
                    help='weight_decay.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout of set transformer.')
if __name__ == '__main__':
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

    train_loader, valid_loader, test_loader, _, _, _, _ = load_data(
        args.batch_size, args.suffix
    )
    if args.cuda:
        predictor.cuda()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)
    loss_fn = nn.MSELoss()
    # Training loop
    best_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        predictor.train()
        epoch_loss = []
        t_start = time.time()
        for batch_idx, (data, relations) in enumerate(train_loader):
            if args.cuda:
                data = data.cuda()
                relations=relations.cuda()

            # mask_tl, mask_br, mask_others=create_masks(args.num_atoms, args.num_visible)
            adj = unflatten_adj_matrix(relations, num_nodes=args.num_atoms)
            # adj[mask_br.unsqueeze_(0).expand(adj.shape[0], -1, -1)]=1
            # adj[mask_others.unsqueeze_(0).expand(adj.shape[0], -1, -1)]=1
            # adj=block_fill(adj,args.num_visible)

            visible_data = data[:, :args.num_visible, :, :]  # [B, V, T, D]
            true_hidden = data[:, args.num_visible:, :, :]   # [B, H, T, D]
            # perm = torch.randperm(args.num_hidden)
            # true_hidden = true_hidden[:, perm, :, :]
            # print(visible_data.shape)
            if args.use_gt_struc and (args.model_type == 'gnn' or args.model_type =='gat' or args.model_type =='rnn'or args.model_type =='ssb'):
                pred_hidden = predictor(visible_data,adj)
            else:
                pred_hidden = predictor(visible_data)
            pred_hidden_aligned,_ = hungarian_match(pred_hidden, true_hidden)
            loss = loss_fn(pred_hidden_aligned, true_hidden)
            # loss = loss_fn(pred_hidden, true_hidden)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
        scheduler.step()
        print(f"Epoch {epoch:03d} | Train Loss: {np.mean(epoch_loss):.6f} | Time: {time.time() - t_start:.2f}s")

        # Optional: validation
        predictor.eval()
        val_loss = []
        val_loss_unaligned=[]
        with torch.no_grad():
            for data, relations in valid_loader:
                if args.cuda:
                    data = data.cuda()
                    relations = relations.cuda()
                adj = unflatten_adj_matrix(relations, num_nodes=args.num_atoms)
                visible_data = data[:, :args.num_visible, :, :]
                true_hidden = data[:, args.num_visible:, :, :]
                # perm = torch.randperm(args.num_hidden)
                # true_hidden = true_hidden[:, perm, :, :]

                if args.use_gt_struc and (args.model_type == 'gnn' or args.model_type =='gat' or args.model_type =='rnn'or args.model_type =='ssb'):
                    pred_hidden = predictor(visible_data, adj)
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
            torch.save(predictor.state_dict(), os.path.join(args.save_folder, args.model_type+"predictor_best.pth"))
            print("Saved best model.")
    print("Training finished.")
