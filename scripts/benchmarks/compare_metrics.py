import json
import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Configura stdout para UTF-8 para evitar problemas de codificacao no Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 Compare Statistics Evaluator Tool")
    parser.add_argument(
        "--dir1",
        type=str,
        default=r"notebooks/testing/test-setup-06-frames",
        help="Caminho para o primeiro diretorio de teste (ex: test-setup-06-frames)"
    )
    parser.add_argument(
        "--dir2",
        type=str,
        default=r"notebooks/testing/test-setup-07-heads-counting",
        help="Caminho para o segundo diretorio de teste (ex: test-setup-07-heads-counting)"
    )
    parser.add_argument(
        "--name1",
        type=str,
        default="Setup 06 (Generic Pose)",
        help="Nome amigavel para identificar o diretorio 1"
    )
    parser.add_argument(
        "--name2",
        type=str,
        default="Setup 07 (Head Detection)",
        help="Nome amigavel para identificar o diretorio 2"
    )
    return parser.parse_args()

def load_metrics(dir_path):
    csv_path = Path(dir_path) / "output" / "frame_counts.csv"
    json_path = Path(dir_path) / "output" / "summary.json"
    
    if not csv_path.exists():
        return None, f"Arquivo CSV nao encontrado em {csv_path}"
    if not json_path.exists():
        return None, f"Arquivo JSON nao encontrado em {json_path}"
        
    try:
        # Carrega JSON
        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        # Carrega CSV
        df = pd.read_csv(csv_path)
        # Identifica a coluna correta de contagem
        count_col = None
        for col in ['people_count', 'count', 'count_people']:
            if col in df.columns:
                count_col = col
                break
        if count_col is None:
            # Seleciona a ultima coluna se nao encontrar nomes conhecidos
            count_col = df.columns[-1]
            
        counts = df[count_col]
        
        # Calcula metricas
        mean_val = counts.mean()
        std_val = counts.std()
        median_val = counts.median()
        min_val = counts.min()
        max_val = counts.max()
        cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
        
        # Ruido temporal (estabilidade)
        diffs = counts.diff().abs().dropna()
        mean_abs_diff = diffs.mean()
        flicker_ratio = (diffs > 3).mean() * 100
        
        metrics = {
            "app_name": summary.get("app", "N/A"),
            "frames": len(df),
            "elapsed_sec": summary.get("elapsed_sec", 0.0),
            "fps": summary.get("fps_processed", 0.0),
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "min": min_val,
            "max": max_val,
            "cv": cv,
            "abs_diff": mean_abs_diff,
            "flicker": flicker_ratio
        }
        return metrics, None
    except Exception as e:
        return None, str(e)

def main():
    args = parse_args()
    
    m1, err1 = load_metrics(args.dir1)
    m2, err2 = load_metrics(args.dir2)
    
    if err1:
        print(f"Erro ao ler Diretorio 1 ({args.dir1}): {err1}")
        sys.exit(1)
    if err2:
        print(f"Erro ao ler Diretorio 2 ({args.dir2}): {err2}")
        sys.exit(1)
        
    print("\n" + "="*80)
    print(f"{'COMPARACAO METODOLOGICA E ESTATISTICA':^80}")
    print("="*80)
    print(f"{'Metrica / Estatistica':<35} | {args.name1:<20} | {args.name2:<20}")
    print("-"*80)
    
    print(f"{'Identificador (App)':<35} | {m1['app_name'][:20]:<20} | {m2['app_name'][:20]:<20}")
    print(f"{'Frames Processados':<35} | {m1['frames']:<20} | {m2['frames']:<20}")
    print(f"{'Tempo de Execucao (segundos)':<35} | {m1['elapsed_sec']:<20.2f} | {m2['elapsed_sec']:<20.2f}")
    print(f"{'Throughput (FPS)':<35} | {m1['fps']:<20.2f} | {m2['fps']:<20.2f}")
    print(f"{'Media de Contagem':<35} | {m1['mean']:<20.2f} | {m2['mean']:<20.2f}")
    print(f"{'Desvio Padrao':<35} | {m1['std']:<20.2f} | {m2['std']:<20.2f}")
    print(f"{'Mediana':<35} | {m1['median']:<20.1f} | {m2['median']:<20.1f}")
    print(f"{'Contagem Minima / Maxima':<35} | {int(m1['min'])}/{int(m1['max']):<15} | {int(m2['min'])}/{int(m2['max']):<15}")
    print(f"{'Coeficiente de Variacao (CV)':<35} | {m1['cv']:<19.2f}% | {m2['cv']:<19.2f}%")
    print(f"{'Ruido Temporal Medio (Frame-a-Frame)':<35} | {m1['abs_diff']:<20.4f} | {m2['abs_diff']:<20.4f}")
    print(f"{'Taxa de Instabilidade Severa (>3)':<35} | {m1['flicker']:<19.2f}% | {m2['flicker']:<19.2f}%")
    
    print("="*80)
    
    # Exibe comparacao de ganhos
    if m1['fps'] > 0 and m2['fps'] > 0:
        speedup = m2['fps'] / m1['fps']
        print(f"Ganho de performance (Throughput): {speedup:.2f}x ({(speedup - 1) * 100:.1f}% mais rapido)")
    
    if m1['abs_diff'] > 0 and m2['abs_diff'] > 0:
        noise_reduction = (1 - (m2['abs_diff'] / m1['abs_diff'])) * 100
        print(f"Reducao de ruido temporal frame-a-frame: {noise_reduction:.2f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
