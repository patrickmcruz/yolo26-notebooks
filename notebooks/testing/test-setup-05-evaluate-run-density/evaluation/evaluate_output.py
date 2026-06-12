import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuracao de caminhos relativos ao projeto test-setup-05
# Este script esta em: test-setup-05-evaluate-run-density/evaluation/evaluate_output.py
# A pasta output esta em: test-setup-05-evaluate-run-density/output
CURRENT_DIR = Path(__file__).parent.resolve()  # .../evaluation/
PROJECT_ROOT = CURRENT_DIR.parent  # .../test-setup-05-evaluate-run-density/
TARGET_OUTPUT_DIR = str(PROJECT_ROOT / "output")

CSV_PATH = os.path.join(TARGET_OUTPUT_DIR, "frame_counts.csv")
JSON_PATH = os.path.join(TARGET_OUTPUT_DIR, "summary.json")
SNAPSHOTS_DIR = os.path.join(TARGET_OUTPUT_DIR, "snapshots")

def evaluate_metrics():
    """
    Carrega os dados gerados pelo modelo YOLO26-Pose (CSV, JSON e Snapshots)
    e gera relatórios analíticos de consistência e distribuição temporal.
    """
    print("=" * 60)
    print("      RELATÓRIO DE AVALIAÇÃO DE CONTAGEM - YOLO26-POSE")
    print("=" * 60)
    print(f"Buscando saídas locais em: {TARGET_OUTPUT_DIR}\n")
    
    # 1. Carregar e Analisar o Summary JSON
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r") as f:
                summary = json.load(f)
            
            print("[INFO] Resumo de Execacao (Summary):")
            
            # Handle different data types safely
            video_path = summary.get('video_path', 'N/A')
            total_frames = summary.get('total_frames', 'N/A')
            avg_count = summary.get('avg_count', 'N/A')
            max_count = summary.get('max_count', 'N/A')
            elapsed_time = summary.get('elapsed_time_seconds', 'N/A')
            fps = summary.get('fps', 'N/A')
            
            print(f" - Video Processado: {video_path}")
            print(f" - Total de Frames Analisados: {total_frames}")
            
            # Only format numbers if they are actually numbers
            if isinstance(avg_count, (int, float)):
                print(f" - Media de Pessoas por Frame: {avg_count:.2f}")
            else:
                print(f" - Media de Pessoas por Frame: {avg_count}")
            
            print(f" - Pico de Publico Detectado: {max_count} pessoas")
            
            if isinstance(elapsed_time, (int, float)):
                print(f" - Tempo de Processamento: {elapsed_time:.2f} segundos")
            else:
                print(f" - Tempo de Processamento: {elapsed_time} segundos")
            
            if isinstance(fps, (int, float)):
                print(f" - Velocidade de Inferencia: {fps:.2f} FPS")
            elif fps != 'N/A':
                print(f" - Velocidade de Inferencia: {fps} FPS")
        except Exception as e:
            print(f"[ERRO] Falha ao ler summary.json: {e}")
    else:
        print(f"[AVISO] Arquivo summary.json não encontrado em: {JSON_PATH}")

    # 2. Carregar e Processar a Série Temporal do CSV
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            
            # Identificação inteligente de colunas
            frame_col = 'frame' if 'frame' in df.columns else df.columns[0]
            count_col = 'count' if 'count' in df.columns else df.columns[1]
            
            print(f"\n[INFO] Série Temporal de Contagem ({len(df)} frames):")
            
            # Estatísticas Descritivas
            counts = df[count_col]
            print(f" - Média de contagem: {counts.mean():.2f} ± {counts.std():.2f}")
            print(f" - Mediana: {counts.median():.1f}")
            print(f" - Coeficiente de Variação (CV): {(counts.std() / counts.mean()) * 100:.2f}%")
            
            # 3. Análise de Ruído Temporal (Flickering)
            diffs = counts.diff().abs().dropna()
            flicker_ratio = (diffs > 3).mean() * 100  # Mudanças bruscas de mais de 3 pessoas consecutivas
            print(f" - Taxa de Ruído Temporal (Instabilidade): {flicker_ratio:.2f}%")

            # 4. Geração de Gráficos Analíticos com Matplotlib e Seaborn
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            plt.figure(figsize=(15, 6))
            
            # Linha temporal
            plt.subplot(1, 2, 1)
            plt.plot(df[frame_col], counts, color='#1f77b4', alpha=0.8, label='Contagem Bruta')
            rolling_mean = counts.rolling(window=30, center=True).mean()
            plt.plot(df[frame_col], rolling_mean, color='#d62728', linewidth=2, label='Tendência (Média Móvel)')
            plt.title('Evolução Temporal da Contagem de Público')
            plt.xlabel('Número do Frame')
            plt.ylabel('Quantidade de Pessoas')
            plt.legend()
            
            # Histograma de densidade
            plt.subplot(1, 2, 2)
            sns.histplot(counts, kde=True, color='#2ca02c', bins=20)
            plt.title('Distribuição da Densidade de Público')
            plt.xlabel('Pessoas por Frame')
            plt.ylabel('Frequência (Frames)')
            
            plt.tight_layout()
            
            # Salva o grafico resultante na raiz do diretorio 05 (test-setup-05-evaluate-run-density)
            chart_path = str(PROJECT_ROOT / "analise_estatistica.png")
            plt.savefig(chart_path, dpi=300)
            print(f"\n[SUCESSO] Grafico analitico salvo em: {chart_path}")
            plt.close()
            
        except Exception as e:
            print(f"[ERRO] Falha ao processar o CSV: {e}")
    else:
        print(f"[AVISO] Arquivo frame_counts.csv não encontrado em: {CSV_PATH}")

    # 5. Avaliacao Visual Qualitativa de Snapshots
    if os.path.exists(SNAPSHOTS_DIR):
        snapshots = [f for f in os.listdir(SNAPSHOTS_DIR) if f.endswith(('.jpg', '.png'))]
        print(f"\n[INFO] Diretorio de Snapshots ({len(snapshots)} capturas):")
        print(f" - Caminho: {SNAPSHOTS_DIR}")
        if len(snapshots) > 0:
            print("   Exemplos disponiveis para auditoria manual:")
            for s in snapshots[:3]:
                print(f"     - {s}")
    else:
        print(f"[AVISO] Pasta de snapshots nao encontrada em: {SNAPSHOTS_DIR}")

if __name__ == "__main__":
    evaluate_metrics()