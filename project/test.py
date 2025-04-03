import os
import sys
import json
import argparse
import numpy as np

import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import neurokit2 as nk
import pywt

###########################################################
# Утилиты: remove_baseline, segment_ecg_by_rpeaks, map_symbol_to_class
###########################################################
def remove_baseline(signal, wavelet='db6'):
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    coeffs[0] = np.zeros_like(coeffs[0])
    rec = pywt.waverec(coeffs, wavelet)
    return rec[:len(signal)]

def segment_ecg_by_rpeaks(ecg_signal, r_idx, segment_size=256):
    half = segment_size // 2
    start = r_idx - half
    end   = r_idx + half
    if start < 0 or end > len(ecg_signal):
        return None
    return ecg_signal[start:end]

def map_symbol_to_class(symbol):
    """
    Пример 5-классной схемы AAMI:
      0 -> Normal (N,L,R,e,j)
      1 -> Supraventricular (A,a,J,S)
      2 -> Ventricular (V,E)
      3 -> Fusion (F)
      4 -> Unknown/Other (/, f, Q, ? ...)
    Подправьте под нужную задачу, если иная.
    """
    if symbol in ['N','L','R','e','j']:
        return 0
    if symbol in ['A','a','J','S']:
        return 1
    if symbol in ['V','E']:
        return 2
    if symbol in ['F']:
        return 3
    return 4


###########################################################
# Датасет для классификации (beat-by-beat), но только для теста
###########################################################
class ECGClassificationDataset(Dataset):
    def __init__(self, record_ids, segment_size=256, wavelet='db6', channel=0):
        super().__init__()
        self.segments = []
        self.labels   = []

        for rid in record_ids:
            record = wfdb.rdrecord(rid)
            fs = record.fs if hasattr(record, 'fs') else 360
            ecg_raw = record.p_signal[:, channel]

            ecg_clean = remove_baseline(ecg_raw, wavelet=wavelet)
            ecg_preproc = nk.ecg_clean(ecg_clean, sampling_rate=fs)

            ann = wfdb.rdann(rid, 'atr')
            # Перебираем R-пики
            for peak_idx, symbol in zip(ann.sample, ann.symbol):
                c = map_symbol_to_class(symbol)
                seg = segment_ecg_by_rpeaks(ecg_clean, peak_idx, segment_size)
                if seg is not None:
                    self.segments.append(seg.astype(np.float32))
                    self.labels.append(c)

        self.segments = np.array(self.segments)  # (N, segment_size)
        self.labels   = np.array(self.labels)    # (N, )
        print(f"[ECGClassificationDataset] total beats={len(self.segments)} from {record_ids}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x = self.segments[idx]  # shape=(segment_size,)
        y = self.labels[idx]
        x_tensor = torch.from_numpy(x).unsqueeze(0)  # shape=(1, segment_size)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor


###########################################################
# Модель Net1D (должна совпадать с обучающей!)
###########################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, groups=groups)
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        in_len = x.shape[-1]
        out_len= (in_len + self.stride - 1)//self.stride
        p = max(0, (out_len-1)*self.stride + self.kernel_size - in_len)
        left = p//2
        right= p-left
        x = F.pad(x,(left,right),'constant',0)
        x = self.conv(x)
        return x

class MyMaxPool1dPadSame(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size)
        self.kernel_size=kernel_size
    def forward(self, x):
        in_len= x.shape[-1]
        p= max(0,self.kernel_size-1)
        left= p//2
        right= p-left
        x= F.pad(x,(left,right),'constant',0)
        x= self.pool(x)
        return x

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

        self.mid_ch= int(self.out_channels*self.ratio)

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
        self.conv3=MyConv1dPadSame(self.mid_ch,self.out_channels,1,1)

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

        # SE
        se= out.mean(dim=-1)
        se= self.se_fc1(se)
        se= self.se_act(se)
        se= self.se_fc2(se)
        se= torch.sigmoid(se)
        out= torch.einsum('bcl,bc->bcl', out,se)

        if self.downsample:
            identity= self.pool(identity)

        if self.out_channels!=self.in_channels:
            ch_diff= self.out_channels-self.in_channels
            left=ch_diff//2
            right=ch_diff-left
            identity= identity.transpose(-1,-2)
            identity= F.pad(identity,(left,right),'constant',0)
            identity= identity.transpose(-1,-2)

        out= out+identity
        return out

class BasicStage(nn.Module):
    def __init__(self,in_channels,out_channels,ratio,kernel_size,stride,groups,
                 i_stage,m_blocks,use_bn=True,use_do=True,dropout_p=0.2):
        super().__init__()
        self.blocks= nn.ModuleList()
        for i_block in range(m_blocks):
            is_first_block=(i_stage==0 and i_block==0)
            ds=(i_block==0)
            tmp_in=in_channels if i_block==0 else out_channels
            block= BasicBlock(tmp_in,out_channels,ratio,kernel_size,stride,groups,
                              downsample=ds,is_first_block=is_first_block,
                              use_bn=use_bn,use_do=use_do,dropout_p=dropout_p)
            self.blocks.append(block)

    def forward(self,x):
        out=x
        for block in self.blocks:
            out= block(out)
        return out

class Net1D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 base_filters=64,
                 ratio=1.0,
                 filter_list=[64,160,160,400],
                 m_blocks_list=[2,2,2,2],
                 kernel_size=16,
                 stride=2,
                 groups_width=16,
                 n_classes=5,    # 5-class
                 use_bn=True,
                 use_do=True,
                 dropout_p=0.2):
        super().__init__()
        self.first_conv= MyConv1dPadSame(in_channels,base_filters,kernel_size,stride=2)
        self.first_bn= nn.BatchNorm1d(base_filters)
        self.first_act= Swish()

        self.stages= nn.ModuleList()
        in_ch= base_filters
        for i_stage, out_ch in enumerate(filter_list):
            m_blocks= m_blocks_list[i_stage]
            stage= BasicStage(in_ch,out_ch,ratio,kernel_size,stride,out_ch//groups_width,
                              i_stage,m_blocks,use_bn=use_bn,use_do=use_do,dropout_p=dropout_p)
            self.stages.append(stage)
            in_ch= out_ch

        self.fc= nn.Linear(in_ch,n_classes)

    def forward(self,x):
        out= self.first_conv(x)
        out= self.first_bn(out)
        out= self.first_act(out)
        for stage in self.stages:
            out= stage(out)
        out= out.mean(dim=-1)
        out= self.fc(out)   # (B, n_classes)
        return out

###########################################################
# Основная функция main()
###########################################################
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default="model_sl_01.pth",
                        help="Path to final classification model .pth")
    parser.add_argument("--records_json", type=str, default="cl_test_records.json",
                        help="JSON-file with list of test record IDs")
    parser.add_argument("--outfile", type=str, default="results_test.txt",
                        help="File to save metrics/CM")
    parser.add_argument("--n_classes", type=int, default=5,
                        help="Number of classification classes (ex. 5 for AAMI scheme)")
    parser.add_argument("--segment_size", type=int, default=256,
                        help="Segment length around R-peak")
    args= parser.parse_args()

    # 1) Загружаем test_records
    if not os.path.isfile(args.records_json):
        print(f"Error: no such file {args.records_json}")
        sys.exit(1)
    with open(args.records_json,"r") as f:
        test_records= json.load(f)
    print("Test records:", test_records)

    # 2) Делаем датасет
    test_ds= ECGClassificationDataset(record_ids=test_records,
                                      segment_size=args.segment_size,
                                      wavelet='db6', channel=0)
    test_loader= DataLoader(test_ds,batch_size=64,shuffle=False)

    # 3) Создаём модель Net1D (те же параметры, кроме n_classes=...)
    model= Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1.0,
        filter_list=[64,160,160,400],
        m_blocks_list=[2,2,2,2],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=args.n_classes,
        use_bn=True,
        use_do=True,
        dropout_p=0.2
    )
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4) Загружаем веса
    if not os.path.isfile(args.model_ckpt):
        print(f"Error: checkpoint {args.model_ckpt} not found!")
        sys.exit(1)
    state_dict= torch.load(args.model_ckpt,map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded model from {args.model_ckpt}")

    # 5) Тестирование
    model.eval()
    preds_all=[]
    trues_all=[]
    with torch.no_grad():
        for (x_batch,y_batch) in test_loader:
            x_batch= x_batch.to(device)
            y_batch= y_batch.to(device)
            logits= model(x_batch)
            preds= torch.argmax(logits,dim=1).cpu().numpy()
            trues= y_batch.cpu().numpy()
            preds_all.append(preds)
            trues_all.append(trues)
    preds_all= np.concatenate(preds_all)
    trues_all= np.concatenate(trues_all)

    # 6) Метрики
    test_acc= accuracy_score(trues_all,preds_all)
    test_prec= precision_score(trues_all,preds_all,average='macro',zero_division=0)
    test_rec= recall_score(trues_all,preds_all,average='macro',zero_division=0)
    test_f1= f1_score(trues_all,preds_all,average='macro',zero_division=0)
    cmat= confusion_matrix(trues_all,preds_all)

    # 7) Сохраняем в outfile
    lines=[]
    lines.append("=== TEST METRICS ===\n")
    lines.append(f"Total test beats={len(test_ds)}\n")
    lines.append(f"Accuracy={test_acc:.3f}\n")
    lines.append(f"Precision(macro)={test_prec:.3f}\n")
    lines.append(f"Recall(macro)={test_rec:.3f}\n")
    lines.append(f"F1(macro)={test_f1:.3f}\n")
    lines.append("Confusion Matrix:\n")
    lines.append(str(cmat)+"\n")

    with open(args.outfile,"w") as f:
        f.writelines(lines)

    # Также печатаем на экран
    print("===== TEST RESULTS =====")
    for ln in lines:
        print(ln,end='')

if __name__=="__main__":
    main()
