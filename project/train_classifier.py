import os
import sys
import json
import random
import argparse

import numpy as np
import wfdb
import pywt
import neurokit2 as nk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import wandb

##################################################
# 1) Установка seeds для воспроизводимости
##################################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Для детерминизма можно включить:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


##################################################
# 2) Remove baseline
##################################################
def remove_baseline(signal, wavelet='db6'):
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    coeffs[0] = np.zeros_like(coeffs[0])
    filtered_signal = pywt.waverec(coeffs, wavelet)
    return filtered_signal[:len(signal)]


##################################################
# 3) segment_ecg_by_rpeaks
##################################################
def segment_ecg_by_rpeaks(ecg_signal, r_idx, segment_size=256):
    half = segment_size//2
    start = r_idx - half
    end   = r_idx + half
    if start<0 or end>len(ecg_signal):
        return None
    return ecg_signal[start:end]


##################################################
# 4) Маппинг символа (N, V, A, ...) -> класс (пример)
##################################################
def map_symbol_to_class(symbol):
    # Пример 5-классного AAMI
    # N (N,L,R,e,j), S (A,a,J,S), V (V,E), F (F), Q (прочее)
    if symbol in ['N','L','R','e','j']:
        return 0
    if symbol in ['A','a','J','S']:
        return 1
    if symbol in ['V','E']:
        return 2
    if symbol in ['F']:
        return 3
    return 4  # всё остальное


##################################################
# 5) Датасет для Классификации R-пиков
##################################################
class ECGClassificationDataset(Dataset):
    def __init__(self, record_ids, segment_size=256, wavelet='db6', channel=0):
        super().__init__()
        self.segments = []
        self.labels   = []
        self.segment_size = segment_size

        for rid in record_ids:
            record = wfdb.rdrecord(rid)
            fs = record.fs if hasattr(record, 'fs') else 360
            ecg_raw = record.p_signal[:, channel]

            ecg_clean = remove_baseline(ecg_raw, wavelet=wavelet)
            ecg_preproc = nk.ecg_clean(ecg_clean, sampling_rate=fs)

            ann = wfdb.rdann(rid, 'atr')  # аннотации
            for peak_idx, symbol in zip(ann.sample, ann.symbol):
                c = map_symbol_to_class(symbol)
                seg = segment_ecg_by_rpeaks(ecg_clean, peak_idx, segment_size)
                if seg is not None:
                    self.segments.append(seg.astype(np.float32))
                    self.labels.append(c)

        self.segments = np.array(self.segments)  # (N, segment_size)
        self.labels   = np.array(self.labels)    # (N,)
        print(f"[ECGClassificationDataset] total beats={len(self.segments)} from records={record_ids}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x = self.segments[idx]  # shape=(segment_size,)
        y = self.labels[idx]
        # Возвращаем (1,L), label
        x_tensor = torch.from_numpy(x).unsqueeze(0)  # (1, segment_size)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor


##################################################
# 6) Архи (Net1D) - тот же, но n_classes может быть
##################################################
class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, groups=groups)
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        in_len = x.shape[-1]
        out_len= (in_len + self.stride -1)//self.stride
        p = max(0, (out_len-1)*self.stride + self.kernel_size - in_len)
        left = p//2
        right= p-left
        x = F.pad(x, (left,right), "constant", 0)
        x = self.conv(x)
        return x

class MyMaxPool1dPadSame(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.kernel_size=kernel_size
        self.pool=nn.MaxPool1d(kernel_size)
    def forward(self,x):
        in_len= x.shape[-1]
        p=max(0,self.kernel_size-1)
        left=p//2
        right=p-left
        x=F.pad(x,(left,right),"constant",0)
        x=self.pool(x)
        return x

class Swish(nn.Module):
    def forward(self,x):
        return x*torch.sigmoid(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, kernel_size, stride, groups,
                 downsample, is_first_block=False, use_bn=True, use_do=True, dropout_p=0.2):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.ratio=ratio
        self.kernel_size=kernel_size
        self.groups=groups
        self.downsample=downsample
        self.stride=stride if downsample else 1
        self.is_first_block=is_first_block
        self.use_bn=use_bn
        self.use_do=use_do

        self.mid_ch=int(self.out_channels*self.ratio)

        self.bn1=nn.BatchNorm1d(in_channels)
        self.act1=Swish()
        self.do1=nn.Dropout(p=dropout_p)
        self.conv1=MyConv1dPadSame(in_channels,self.mid_ch,1,1)

        self.bn2=nn.BatchNorm1d(self.mid_ch)
        self.act2=Swish()
        self.do2=nn.Dropout(p=dropout_p)
        self.conv2=MyConv1dPadSame(self.mid_ch,self.mid_ch,kernel_size,self.stride,groups=self.groups)

        self.bn3=nn.BatchNorm1d(self.mid_ch)
        self.act3=Swish()
        self.do3=nn.Dropout(p=dropout_p)
        self.conv3=MyConv1dPadSame(self.mid_ch,out_channels,1,1)

        # SE
        self.se_fc1=nn.Linear(out_channels,out_channels//2)
        self.se_fc2=nn.Linear(out_channels//2,out_channels)
        self.se_act=Swish()

        if self.downsample:
            self.pool=MyMaxPool1dPadSame(self.stride)

    def forward(self,x):
        identity=x
        out=x
        if not self.is_first_block:
            if self.use_bn:
                out=self.bn1(out)
            out=self.act1(out)
            if self.use_do:
                out=self.do1(out)
        out=self.conv1(out)
        if self.use_bn:
            out=self.bn2(out)
        out=self.act2(out)
        if self.use_do:
            out=self.do2(out)
        out=self.conv2(out)

        if self.use_bn:
            out=self.bn3(out)
        out=self.act3(out)
        if self.use_do:
            out=self.do3(out)
        out=self.conv3(out)

        se=out.mean(dim=-1)
        se=self.se_fc1(se)
        se=self.se_act(se)
        se=self.se_fc2(se)
        se=torch.sigmoid(se)
        out=torch.einsum('bcl,bc->bcl', out,se)

        if self.downsample:
            identity=self.pool(identity)

        if self.out_channels!=self.in_channels:
            ch_diff=self.out_channels-self.in_channels
            left=ch_diff//2
            right=ch_diff-left
            identity=identity.transpose(-1,-2)
            identity=F.pad(identity,(left,right),"constant",0)
            identity=identity.transpose(-1,-2)
        out=out+identity
        return out

class BasicStage(nn.Module):
    def __init__(self,in_channels,out_channels,ratio,kernel_size,stride,groups,
                 i_stage,m_blocks,use_bn=True,use_do=True,dropout_p=0.2):
        super().__init__()
        self.blocks=nn.ModuleList()
        for i_block in range(m_blocks):
            is_first_block=(i_stage==0 and i_block==0)
            ds=(i_block==0)
            tmp_in=in_channels if i_block==0 else out_channels
            block=BasicBlock(tmp_in,out_channels,ratio,kernel_size,stride,groups,
                             downsample=ds, is_first_block=is_first_block,
                             use_bn=use_bn, use_do=use_do, dropout_p=dropout_p)
            self.blocks.append(block)

    def forward(self,x):
        out=x
        for block in self.blocks:
            out=block(out)
        return out

class Net1D(nn.Module):
    def __init__(self,in_channels,base_filters,ratio,filter_list,m_blocks_list,
                 kernel_size,stride,groups_width,n_classes,
                 use_bn=True,use_do=True,dropout_p=0.2):
        super().__init__()
        self.first_conv=MyConv1dPadSame(in_channels,base_filters,kernel_size,stride=2)
        self.first_bn=nn.BatchNorm1d(base_filters)
        self.first_act=Swish()

        self.stages=nn.ModuleList()
        in_ch=base_filters
        for i_stage,out_ch in enumerate(filter_list):
            m_blocks=m_blocks_list[i_stage]
            stage=BasicStage(in_ch,out_ch,ratio,kernel_size,stride,out_ch//groups_width,
                             i_stage,m_blocks,use_bn=use_bn,use_do=use_do,dropout_p=dropout_p)
            self.stages.append(stage)
            in_ch=out_ch

        self.fc=nn.Linear(in_ch,n_classes)

    def forward(self,x):
        out=self.first_conv(x)
        out=self.first_bn(out)
        out=self.first_act(out)
        for stage in self.stages:
            out=stage(out)
        out=out.mean(dim=-1)   # GAP
        out=self.fc(out)       # (B,n_classes)
        return out


###############################################################
# 7) train_classifier: единая функция, режим sl или ssl
###############################################################
def train_classifier_wandb(
    record_ids,            # список записей (строки)
    mode="sl",             # "sl" - train from scratch, "ssl" - load pretrained + freeze
    pretrained_path="",    # путь к encoder SSL, если mode="ssl"
    n_classes=5,
    segment_size=256,
    wavelet='db6',
    channel=0,
    batch_size=64,
    epochs=51,
    val_ratio=0.2,
    checkpoint_interval=3,
    freeze_encoder=True    # если mode=ssl, freeze или нет
):
    """
    Аргумент mode:
      - "sl": обучаем всё "с нуля"
      - "ssl": используем энкодер из SSL (encoder_ssl.pth). Замораживаем его слои, 
               а финальный fc (кол-во классов) обучаем.
               (или можно freeze=False, тогда fine-tune все слои).
    
    Возвращает обученную модель.
    """

    wandb.init(
        project="ecg-classification",
        name=f"classifier_{mode}",
        config={
            "mode": mode,
            "n_classes": n_classes,
            "segment_size": segment_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "val_ratio": val_ratio,
            "freeze_encoder": freeze_encoder
        }
    )

    # 1) Создаём датасет
    full_ds = ECGClassificationDataset(
        record_ids=record_ids,
        segment_size=segment_size,
        wavelet=wavelet,
        channel=channel
    )
    n_total=len(full_ds)
    n_val=int(val_ratio*n_total)
    n_train=n_total-n_val

    set_seed(42)
    
    train_ds, val_ds = random_split(full_ds,[n_train,n_val])

    print(f"Train dataset: {len(train_ds)} beats, Val dataset: {len(val_ds)} beats.")
    # 2) DataLoader
    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size,shuffle=False)

    # 3) Создаём Net1D
    #    Если "ssl" - нужно n_classes=128 в Net1D, чтобы загрузить веса
    #    потом заменить fc -> => 5
    if mode=="ssl":
        # Создаём модель c n_classes=128, загрузим энкодерные веса
        model_ssl = Net1D(
            in_channels=1,
            base_filters=64,
            ratio=1.0,
            filter_list=[64,160,160,400],
            m_blocks_list=[2,2,2,2],
            kernel_size=16,
            stride=2,
            groups_width=16,
            n_classes=128,   # как в SSL
            use_bn=True,
            use_do=True,
            dropout_p=0.2
        )
        # грузим
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_ssl.to(device)
        if os.path.isfile(pretrained_path):
            print(f"Loading SSL encoder from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=device)
            model_ssl.load_state_dict(state_dict, strict=True)
        else:
            print(f"Error: pretrained_path={pretrained_path} not found.")
            sys.exit(1)

        # Теперь заменим финальный fc на новую "head" => n_classes
        # Сохраним старый in_ch
        # find out in_ch from model_ssl.fc.in_features
        in_ch = model_ssl.fc.in_features
        # создаём новую fc
        new_fc = nn.Linear(in_ch, n_classes)
        model_ssl.fc = new_fc
        model_ssl.to(device) 
        # model_ssl.fc.weight и bias - новые, остальное - предобученное

        # freeze или нет
        if freeze_encoder:
            # Замораживаем всё, кроме final fc
            for name,param in model_ssl.named_parameters():
                if "fc." not in name:  # если это не финальный слой
                    param.requires_grad = False
        # В итоге trainable будут только параметры fc
        model = model_ssl

    else:
        # mode="sl": train with n_classes from scratch
        model = Net1D(
            in_channels=1,
            base_filters=64,
            ratio=1.0,
            filter_list=[64,160,160,400],
            m_blocks_list=[2,2,2,2],
            kernel_size=16,
            stride=2,
            groups_width=16,
            n_classes=n_classes,  
            use_bn=True,
            use_do=True,
            dropout_p=0.2
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    # 4) Критерий, оптимизатор
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    wandb.watch(model, log="all")

    # 5) Train loop
    for epoch in range(1, epochs+1):
        model.train()
        total_loss_train=0.0
        preds_train = []
        trues_train = []

        for (x_batch,y_batch) in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()

            # собираем предикты
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            trues = y_batch.cpu().numpy()
            preds_train.append(preds)
            trues_train.append(trues)

        train_preds = np.concatenate(preds_train)
        train_trues = np.concatenate(trues_train)
        train_loss  = total_loss_train/len(train_loader)
        train_acc   = accuracy_score(train_trues, train_preds)
        train_prec  = precision_score(train_trues, train_preds, average='macro', zero_division=0)
        train_rec   = recall_score(train_trues, train_preds, average='macro', zero_division=0)
        train_f1    = f1_score(train_trues, train_preds, average='macro', zero_division=0)

        # --- валидируем ---
        model.eval()
        preds_val=[]
        trues_val=[]
        total_loss_val=0.0
        with torch.no_grad():
            for (x_val,y_val) in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                logits= model(x_val)
                loss_val= criterion(logits,y_val)
                total_loss_val += loss_val.item()

                pv= torch.argmax(logits,dim=1).cpu().numpy()
                tv= y_val.cpu().numpy()
                preds_val.append(pv)
                trues_val.append(tv)

        val_preds= np.concatenate(preds_val)
        val_trues= np.concatenate(trues_val)
        val_loss= total_loss_val/len(val_loader)
        val_acc= accuracy_score(val_trues,val_preds)
        val_prec= precision_score(val_trues,val_preds,average='macro',zero_division=0)
        val_rec= recall_score(val_trues,val_preds,average='macro',zero_division=0)
        val_f1= f1_score(val_trues,val_preds,average='macro',zero_division=0)

        # confusion matrix - можно логировать в конце, 
        # а можно раз в эпоху:
        cm= confusion_matrix(val_trues,val_preds)

        print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_f1={train_f1:.3f}, val_f1={val_f1:.3f}")

        # wandb.log
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
            "train_prec": train_prec,"val_prec": val_prec,
            "train_rec": train_rec, "val_rec": val_rec,
            "train_f1": train_f1,   "val_f1": val_f1,
            # confusion matrix можно логировать как wandb.plot.confusion_matrix
            # Для демонстрации:
            "confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                 y_true=val_trues, preds=val_preds,
                 class_names=[0,1,2,3,4]) # или ["N","S","V","F","Q"]
        })

        if epoch%checkpoint_interval==0:
            ckpt_path = f"c:/Users/danii/Desktop/master/kursach/checkpoints_cl_01/model_{mode}_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            # wandb.save(ckpt_path)

    print("Done training classification model!")
    return model

def save_records_json(records_list, filename="test_records.json"):
    with open(filename, "w") as f:
        json.dump(records_list, f, indent=2)

def load_records_json(filename="test_records.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    return data



##########################################################
# 10) Main
##########################################################
if __name__=="__main__":

    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="sl", 
                        help="sl (train from scratch) or ssl (load SSL encoder).")
    parser.add_argument("--pretrained_path", type=str, default="encoder_ssl.pth",
                        help="Path to SSL encoder weights, used if mode=ssl")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="If set, freeze the loaded encoder (ssl mode).")
    parser.add_argument("--records_json", type=str, default="test_records.json",
                        help="Which records to use for classification dataset.")
    # любые другие аргументы
    args = parser.parse_args()

    # 1) Считываем 4 записи из test_records.json
    if not os.path.isfile(args.records_json):
        print(f"Error: file {args.records_json} not found!")
        sys.exit(1)

    with open(args.records_json,"r") as f:
        all_4_records = json.load(f)
    print("Loaded records (4) from", args.records_json, ":", all_4_records)

    # 2) Разделяем их на trainval и test (например, 1 запись в test, 3 – trainval)
    random.shuffle(all_4_records)
    n_total = len(all_4_records)
    n_test = 1  # 1 запись в финальный тест
    test_records = all_4_records[:n_test]
    trainval_records = all_4_records[n_test:]
    print(f"test_records={test_records}, trainval_records={trainval_records}")

    save_records_json(trainval_records, "cl_trainval_records.json")
    save_records_json(test_records, "cl_test_records.json")


    # 3) Запускаем training-функцию (train_classifier_wandb) ТОЛЬКО на trainval_records
    #    внутри неё всё равно есть split на train/val (например, 80/20 по beat-сегментам).
    model = train_classifier_wandb(
        record_ids=trainval_records,
        mode=args.mode,
        pretrained_path=args.pretrained_path,
        n_classes=5, 
        segment_size=256,
        batch_size=64,
        epochs=51,
        val_ratio=0.2,
        checkpoint_interval=3,
        freeze_encoder=args.freeze_encoder
    )

    # 4) Модель обучена. При желании, теперь можно проверить на test_records 
    #    (где 1 запись), прогнав функцию evaluate_model(...).
    #    Или сохранить модель:
    final_ckpt = f"c:/Users/danii/Desktop/master/kursach/checkpoints_cl_01/model_{args.mode}_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    print(f"Saved final model to {final_ckpt}")
