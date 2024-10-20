import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp

from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import sys
import clip
import ftfy
import regex as re
import html
import random
import tempfile
import tokenizers
import json
import random
import math


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        """
        訓練データの総数を返す
        """
        return len(self.captions_tokens)
    
    def pad_tokens(self, item: int):
        """
        特定のindexのキャプショントークン列をpaddingする
        """
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        return tokens

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        """
        特定のindexのキャプショントークン列を返す
        Return 
            clip_tokens:    paddingされたキャプショントークン列
            clip_tokens_77: paddingしていないキャプショントークン列
        """
        clip_tokens = self.pad_tokens(item)
        clip_tokens_77 = self.captions_tokens[item]
        return clip_tokens, clip_tokens_77

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        """
        コンストラクタ
        """
        self.clip_tokenizer = clip.tokenize
        self.prefix_length = 10
        self.max_seq_len = 20
        with open(data_path, 'r') as f:
            self.captions = json.load(f)
        random.shuffle(self.captions)
        self.captions_tokens = [] # 各訓練データのキャプショントークン列
        for caption in self.captions[:]:
            try:
                self.captions_tokens.append(self.clip_tokenizer(caption)[0].long()) # tokenize
            except:
                continue
        print(f"paddingにおける最大長: {self.max_seq_len}")
        print(f"訓練データの保存先: {data_path}")
        print(f"訓練データの総数: {len(self.captions_tokens)}件")

    
class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播（フォワードパス）の計算を定義
        """
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        """
        モデルの構成（レイヤーやパラメータなど）を定義
        """
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):     # 各層を設定
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias)) #線形層
            if i < len(sizes) - 2:
                layers.append(act())        # 活性化関数
        self.model = nn.Sequential(*layers)


class DeCap(nn.Module):

    def __init__(self, prefix_size: int = 512):
        super(DeCap, self).__init__()

        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        with open('./decoder_config.pkl','rb') as f:
            config = pickle.load(f)
        self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]  # GPT-2の埋め込み次元
        self.clip_project = MLP((prefix_size, self.embedding_size))
        
    def forward(self, clip_features, gpt_tokens):
        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        embedding_clip = self.clip_project(clip_features)                   # CLIPの埋め込み次元→GPT-2の埋め込み次元
        embedding_clip = embedding_clip.reshape(-1, 1, self.embedding_size)
        embedding_cat = torch.cat([embedding_clip, embedding_text], dim=1)  # 埋め込みを連結
        out = self.decoder(inputs_embeds=embedding_cat)                     # 次のトークンを生成
        # print(f"clip_features.shape:{clip_features.shape}")
        # print(f"gpt_tokens:{gpt_tokens}")
        # print(f"gpt_tokens.shape:{gpt_tokens.shape}")
        # print(f"embedding_text.shape:{embedding_text.shape}")
        # print(f"embedding_clip.shape:{embedding_clip.shape}")
        # print(f"embedding_clip.shape:{embedding_clip.shape}")
        # print(f"embedding_cat.shape: {embedding_cat.shape}")
        # print(out.logits.shape)
        return out


def save_config(args: argparse.Namespace):
    """
    未使用
    """
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    """
    未使用
    """
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # モデルの重みをCPUにロード
    else:
        print(f"{model_path} is not exist")
    return model, parser
    

def train_decoder(dataset: ClipCocoDataset, args,
          lr: float = 1e-5, warmup_steps: int = 1000, output_dir: str = ".", output_prefix: str = ""):
    """
    デコーダを訓練する
    """
    batch_size = args.bs
    epochs = args.epochs
    print(f"バッチサイズ: {batch_size}")
    print(f"エポック数: {epochs}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('デバイス: GPU' if torch.cuda.is_available() else 'デバイス: CPU')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DeCap()

    clip_model_type = "ViT-B/32"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    clip_model.eval()

    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    # トレーニングループ
    for epoch in range(epochs):
        loss_token_save, ac_save= 0, 0
        sys.stdout.flush()
        print(f">>> Training epoch {epoch}")
        progress = tqdm(total=int(len(train_dataloader) / 10), desc=output_prefix)

        for batch_idx, (clip_tokens, clip_tokens_77) in enumerate(train_dataloader):
            clip_tokens, clip_tokens_77 = clip_tokens.to(device), clip_tokens_77.to(device)
            
            with torch.no_grad():
                feature_text = clip_model.encode_text(clip_tokens_77)
                feature_text /= feature_text.norm(dim=-1, keepdim=True)

            outputs = model(feature_text.float(), clip_tokens)
            logits = outputs
            
            logits = logits.logits

            logits = logits[:,: -1]
            clip_tokens = clip_tokens.flatten()
            logits = logits.reshape(-1, logits.shape[-1])
            
            loss_token = loss_ce(logits, clip_tokens)
            ac = ((logits.argmax(1) == clip_tokens) * (clip_tokens > 0)).sum() / (clip_tokens > 0).sum()
            optimizer.zero_grad()
            loss_all = loss_token
            loss_all.backward()
            optimizer.step()
            scheduler.step()

            # 10バッチごとに損失と精度の平均値を表示
            if (batch_idx + 1) % 10 == 0:
                progress.set_postfix({"loss_token": loss_token_save / 10.0, "acc_token": ac_save / 10.0})
                progress.update()
                loss_token_save, ac_save = 0, 0
            else:
                loss_token_save += loss_token.item()
                ac_save += ac.item()

        # 各エポックの終了時、損失や精度をログファイルに書き込む
        log_dir = './log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_txt_path = './log/' + args.dataset + '.txt'
        with open(log_txt_path, 'a+') as f:
            f.writelines('epoch ' + str(epoch) + ': ' + progress.postfix + '\r\n')
        progress.close()

        # 各エポック（または最終エポック）の終了時、モデルのパラメータを保存
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/doubleMTA_lr1e5_ignore0.pkl')
    parser.add_argument('--out_dir', default='./coco_model')                            # モデル保存先
    parser.add_argument('--prefix', default='./coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--dataset', default='coco', help='coco or cc3m or bookcorpus') # データセット名
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=1)
    parser.add_argument('--prefix_length_clip', type=int, default=1)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.') 
    args = parser.parse_args()
    prefix_length = args.prefix_length

    dataset = ClipCocoDataset('data/'+args.dataset+'_train.json', prefix_length, normalize_prefix=args.normalize_prefix)
    train_decoder(dataset, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    main()
