import argparse
import math
import os
import sys
import numpy as np
import torch
import joblib as jl
import pandas as pd
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from transformers import BertTokenizer, BertModel

# Tenta importar pymo
try:
    from pymo.parsers import BVHParser
    from pymo.writers import BVHWriter
except ImportError:
    print("ERRO: 'pymo' não encontrado. Instale ou verifique o PYTHONPATH.")
    sys.exit(1)

# Limpeza de avisos
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message=".*sklearn.*")

sys.path.append(os.getcwd())
import utils
import utils.train_utils_bert
from utils.data_utils import SubtitleWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_bert_features(text_list, tokenizer, bert_model):
    if not text_list: return torch.zeros((1, 1, 3072)).to(device)
    sentence = " ".join([t[0] for t in text_list])
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs, output_hidden_states=True)
    return torch.cat(outputs.hidden_states[-4:], dim=-1)

def generate_gestures(args, pose_decoder, tokenizer, bert_model, words):
    clip_length = words[-1][2] if len(words) > 0 else 5.0
    pre_seq = torch.zeros((1, args.n_pre_poses, pose_decoder.pose_dim))
    mean_pose = torch.squeeze(torch.Tensor(args.data_mean))
    pre_seq[0, :, :] = mean_pose.repeat(args.n_pre_poses, 1)

    unit_time = args.n_poses / args.motion_resampling_framerate
    stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
    num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1 if clip_length >= unit_time else 1

    out_list = []
    out_poses = None
    
    for i in range(num_subdivision):
        start_time = i * stride_time
        end_time = start_time + unit_time
        word_seq = [w for w in words if w[1] <= end_time and w[2] >= start_time]
        in_text_emb = get_bert_features(word_seq, tokenizer, bert_model)
        
        if i > 0 and out_poses is not None:
            pre_seq[0, :, :] = out_poses.squeeze(0)[-args.n_pre_poses:]
        
        pre_seq = pre_seq.float().to(device)
        words_lengths = torch.LongTensor([in_text_emb.shape[1]]).to(device)
        out_poses = pose_decoder(in_text_emb, words_lengths, pre_seq, None)
        out_seq = out_poses[0, :, :].data.cpu().numpy()

        if len(out_list) > 0:
            last_poses = out_list[-1][-args.n_pre_poses:]
            out_list[-1] = out_list[-1][:-args.n_pre_poses]
            for j in range(len(last_poses)):
                n = len(last_poses)
                out_seq[j] = last_poses[j] * (n - j) / (n + 1) + out_seq[j] * (j + 1) / (n + 1)
        out_list.append(out_seq)

    return np.vstack(out_list) if len(out_list) > 0 else np.zeros((1, pose_decoder.pose_dim))

def process_single_file(tsv_path, pipeline, args, generator, tokenizer, bert_model):
    print(f"\nProcessing: {tsv_path.name}")
    
    # 1. Encontrar o BVH Template
    bvh_folder = tsv_path.parents[1] / 'bvh'
    template_path = bvh_folder / (tsv_path.stem + '.bvh')
    
    # Fallback
    if not template_path.exists():
        template_path = tsv_path.with_suffix('.bvh')
        
    if not template_path.exists():
        # Tenta pegar qualquer um da pasta
        candidates = list(bvh_folder.glob('*.bvh'))
        if candidates: template_path = candidates[0]
        else:
            print(f"  [PULA] BVH Template não encontrado.")
            return

    # 2. Ler Texto
    try:
        transcript = SubtitleWrapper(str(tsv_path)).get()
    except Exception as e:
        print(f"  [ERRO] Falha ao ler TSV: {e}")
        return

    word_list = []
    for wi in range(len(transcript)):
        word_s, word_e = float(transcript[wi][0]), float(transcript[wi][1])
        word = transcript[wi][2].strip()
        word_tokens = word.split()
        for t_i, token in enumerate(word_tokens):
            if len(token)>0:
                new_s = word_s + (word_e - word_s) * t_i / len(word_tokens)
                new_e = word_s + (word_e - word_s) * (t_i + 1) / len(word_tokens)
                word_list.append([token, new_s, new_e])

    # 3. Gerar Movimento (IA)
    out_poses = generate_gestures(args, generator, tokenizer, bert_model, word_list)
    
    # CORTE DE DIMENSÃO (Fix V26)
    if out_poses.shape[1] > 108:
        out_poses = out_poses[:, :108]

    # Suavização
    for i in range(out_poses.shape[1]):
        out_poses[:, i] = savgol_filter(out_poses[:, i], 15, 2)

    # 4. Desnormalizar (Pipeline)
    try:
        reconstructed_list = pipeline.inverse_transform([out_poses])
        pred_df = reconstructed_list[0].values
    except ValueError as ve:
        print(f"  [ERRO] Shape Mismatch: {ve}")
        return
    except Exception as e:
        print(f"  [ERRO] Pipeline falhou: {e}")
        return

    # ====================================================================
    # --- DIAGNÓSTICO E TROCA DE BRAÇOS ---
    # ====================================================================
    print("  [DEBUG] Colunas geradas pelo Pipeline (Amostra):")
    cols = list(pred_df.columns)
    # Imprime as primeiras 10 colunas para ver se os nomes batem
    print(f"    {cols[:5]} ...")
    
    print("  [FIX] Tentando trocar Esquerda <-> Direita canal por canal...")
    
    swap_pairs = [
        ('b_l_shoulder', 'b_r_shoulder'),
        ('b_l_arm',      'b_r_arm'),
        ('b_l_forearm',  'b_r_forearm'),
        ('b_l_hand',     'b_r_hand'),
        ('b_l_wrist',    'b_r_wrist')
    ]

    swapped_count = 0
    for l_bone, r_bone in swap_pairs:
        # Tenta trocar cada canal individualmente (Xrotation, Yrotation, etc)
        # Não espera que todos existam ao mesmo tempo.
        for suffix in ['Xrotation', 'Yrotation', 'Zrotation', 'Xposition', 'Yposition', 'Zposition']:
            l_col = f"{l_bone}_{suffix}"
            r_col = f"{r_bone}_{suffix}"
            
            if l_col in cols and r_col in cols:
                # Efetua a troca
                temp = pred_df[l_col].copy()
                pred_df[l_col] = pred_df[r_col]
                pred_df[r_col] = temp
                swapped_count += 1
                # print(f"    -> Trocado: {l_col} <-> {r_col}")

    if swapped_count == 0:
        print("  [ALERTA] NENHUMA COLUNA FOI TROCADA! Verifique se os nomes dos ossos batem com a lista acima.")
    else:
        print(f"  [SUCESSO] Total de canais trocados: {swapped_count}")

    # ====================================================================

    # 5. Merge com Template
    parser = BVHParser()
    writer = BVHWriter()
    full_skeleton = parser.parse(str(template_path))
    
    n_frames = len(pred_df)
    new_index = pd.to_timedelta(np.arange(n_frames) * full_skeleton.framerate, unit='s')
    final_df = pd.DataFrame(0.0, index=new_index, columns=full_skeleton.values.columns)
    
    # Lógica V26 Original (Mantém o corpo perfeito)
    for col in final_df.columns:
        if col in pred_df.columns:
            final_df[col] = pred_df[col].values
        else:
            final_df[col] = 0.0
            # Fix root height offset se faltar
            if 'b_root' in col and 'Yposition' in col:
                offset_y = 91.5
                if 'b_root' in full_skeleton.skeleton:
                    offset_y = full_skeleton.skeleton['b_root']['offsets'][1]
                final_df[col] = offset_y

    full_skeleton.values = final_df
    
    # Salvar
    save_path = Path('../output/infer_bert_final') 
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / (tsv_path.stem + '_generated.bvh')
    
    with open(out_file, 'w') as f:
        writer.write(full_skeleton, f)
    print(f"  [OK] Salvo: {out_file}")

def main(checkpoint_path, input_path):
    print(f"--- INICIANDO INFERÊNCIA BERT (V36 - DIAGNOSTIC SWAP) ---")
    
    # Carregar Pipeline
    possible_pipe_paths = [Path('../resource/data_pipe.sav'), Path('./data_pipe.sav')]
    pipe_path = next((p for p in possible_pipe_paths if p.exists()), None)
    if not pipe_path:
        print("ERRO CRÍTICO: 'data_pipe.sav' não encontrado.")
        return
    pipeline = jl.load(str(pipe_path))

    # Carregar Modelo
    args, generator, loss_fn, _, out_dim = utils.train_utils_bert.load_checkpoint_and_model(checkpoint_path, device)
    bert_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    bert_model = BertModel.from_pretrained(bert_name).to(device)
    bert_model.eval()

    if input_path.is_dir():
        tsv_files = sorted(list(input_path.glob("*.tsv")))
        print(f"Encontrados {len(tsv_files)} arquivos.")
        for tsv_file in tsv_files:
            process_single_file(tsv_file, pipeline, args, generator, tokenizer, bert_model)
    else:
        process_single_file(input_path, pipeline, args, generator, tokenizer, bert_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("transcript_path", type=Path)
    args = parser.parse_args()

    main(args.ckpt_path, args.transcript_path)