from __future__ import annotations
# Auto-generated from jupyter notebook
import sys

# --- CELL 1 ---
# from __future__ import annotations

import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def find_notebook_dir() -> Path:
    """
    Localiza dinamicamente o diretório ativo do setup 07 (Heads Counting) 
    para carregar o arquivo de configuração data.yaml.
    """
    cwd = Path.cwd().resolve()
    # Mapeamento do diretório do novo sandbox 'test-setup-07-heads-counting'
    target_subdir = Path("notebooks") / "testing" / "test-setup-07-heads-counting"
    alt_subdir = Path("test-setup-07-heads-counting")
    
    candidates = [
        cwd,
        cwd / target_subdir,
        cwd / alt_subdir,
        cwd.parent / target_subdir,
        cwd.parent / alt_subdir,
    ]
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / target_subdir)
        candidates.append(parent / alt_subdir)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "data.yaml").exists():
            return candidate

    raise FileNotFoundError(
        "Não foi possível encontrar o arquivo test-setup-07-heads-counting/data.yaml "
        "a partir do diretório de trabalho atual."
    )


NOTEBOOK_DIR = find_notebook_dir()
CONFIG_PATH = NOTEBOOK_DIR / "data.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG: dict[str, Any] = yaml.safe_load(file) or {}


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (NOTEBOOK_DIR / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_environment(config: dict[str, Any]) -> None:
    path_like_keys = {"YOLO_CONFIG_DIR", "MPLCONFIGDIR", "TORCH_HOME"}
    for key, value in (config.get("environment") or {}).items():
        env_value = resolve_path(value) if key in path_like_keys else value
        if key in path_like_keys:
            ensure_dir(Path(env_value))
        os.environ[str(key)] = str(env_value)


apply_environment(CONFIG)
random.seed(int(CONFIG.get("app", {}).get("seed", 42)))

APP_CONFIG = CONFIG.get("app", {})
if APP_CONFIG.get("install_requirements", False):
    requirements_path = resolve_path(APP_CONFIG.get("requirements_file", "../../requirements.txt"))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])

PATH_CONFIG = CONFIG.get("paths", {})
OUTPUT_CONFIG = CONFIG.get("output", {})

VIDEO_PATH = resolve_path(PATH_CONFIG["video"])
OUTPUT_DIR = ensure_dir(resolve_path(PATH_CONFIG.get("output_dir", "output")))
ANNOTATED_VIDEO_PATH = resolve_path(PATH_CONFIG.get("annotated_video", OUTPUT_DIR / "annotated.mp4"))

# Resolução de arquivos estatísticos
FRAME_COUNTS_CSV = resolve_path(PATH_CONFIG.get("frame_counts_csv", OUTPUT_DIR / "frame_counts.csv"))
SUMMARY_JSON = resolve_path(PATH_CONFIG.get("summary_json", OUTPUT_DIR / "summary.json"))
SNAPSHOTS_DIR = ensure_dir(resolve_path(PATH_CONFIG.get("snapshots_dir", OUTPUT_DIR / "snapshots")))

for path in [ANNOTATED_VIDEO_PATH.parent, FRAME_COUNTS_CSV.parent, SUMMARY_JSON.parent]:
    ensure_dir(path)

print(f"Notebook dir: {NOTEBOOK_DIR}")
print(f"Config: {CONFIG_PATH}")
print(f"Video: {VIDEO_PATH}")
print(f"Output dir: {OUTPUT_DIR}")

# --- CELL 2 ---
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

# Desativa a sincronização síncrona do YOLO para melhorar o throughput de processamento
try:
    from ultralytics.utils import SETTINGS
    SETTINGS.update({"sync": False})
except Exception:
    pass

# 1. Carregamento de Parâmetros do data.yaml (CONFIG foi definido na célula anterior)
RUNTIME_CONFIG = CONFIG.get("runtime", {})
DEVICE = int(RUNTIME_CONFIG.get("device", 0))
REQUIRE_CUDA = bool(RUNTIME_CONFIG.get("require_cuda", True))

# 2. Validação e Configuração do Ambiente CUDA (RTX 4090)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA é obrigatório de acordo com o data.yaml, mas o PyTorch não conseguiu identificar nenhuma GPU ativa."
    )

if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE)
    
    # Ativa/Desativa o uso de TensorFloat32 (TF32) nos Tensor Cores da RTX 4090
    allow_tf32 = bool(RUNTIME_CONFIG.get("allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    
    # Ativa benchmark do cuDNN para encontrar os algoritmos de convolução mais rápidos para o tamanho imgsz do YOLO
    torch.backends.cudnn.benchmark = True

# 3. Configuração de Precisão para Multiplicação de Matrizes (Float32)
# Opções no YAML: "highest" (precisão máxima), "high" (ótimo balanço) ou "medium" (rápido)
precision_mode = RUNTIME_CONFIG.get("torch_float32_matmul_precision", "highest")
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision(precision_mode)

# 4. Relatório de Inicialização do Sistema
print("=" * 60)
print("             SISTEMA DE INICIALIZAÇÃO - YOLO26 RUNTIME")
print("=" * 60)
print(f"Ultralytics:     {ultralytics.__version__}")
print(f"PyTorch:         {torch.__version__}")
print(f"OpenCV:          {cv2.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    free_bytes, total_bytes = torch.cuda.mem_get_info(DEVICE)
    gpu_name = torch.cuda.get_device_name(DEVICE)
    print(f"GPU Ativa:       {gpu_name} (ID: {DEVICE})")
    print(f"VRAM da GPU:     {free_bytes / 1024**3:.2f} / {total_bytes / 1024**3:.2f} GiB livres")
    print(f"Modo TF32:       {'Ativado' if allow_tf32 else 'Desativado'}")
    print(f"Precisão Matmul: {precision_mode}")

# 5. Relatório de Memória RAM do Sistema (Aproveitando os 128GB do setup)
try:
    import psutil
    ram = psutil.virtual_memory()
    print(f"RAM do Sistema:  {ram.available / 1024**3:.2f} / {ram.total / 1024**3:.2f} GiB disponíveis")
except Exception:
    print("RAM do Sistema:  psutil não está disponível para medições.")
print("=" * 60)

# --- CELL 3 ---
def resolve_weight_reference(paths_config: dict[str, Any]) -> str:
    """
    Resolve o caminho dos pesos do modelo. 
    Otimizado para preferir versões compiladas em TensorRT (.engine) se disponíveis,
    garantindo performance máxima na RTX 4090.
    """
    # Alterado o padrão para yolo26x.pt (detecção pura) em vez de pose
    weight_ref = str(paths_config.get("weights", "yolo26x.pt"))
    weight_path = Path(weight_ref).expanduser()

    # Define os nomes de arquivos alternativos para checar por TensorRT (.engine)
    engine_ref = weight_ref.replace(".pt", ".engine") if weight_ref.endswith(".pt") else None

    # 1. Se for um caminho absoluto existente
    if weight_path.is_absolute() and weight_path.exists():
        return str(weight_path.resolve())

    # 2. Busca nos diretórios configurados (Priorizando .engine para aceleração de hardware)
    search_dirs = paths_config.get("weights_search_dirs", [])
    
    # Primeiro passo de busca: tentar encontrar a versão TensorRT (.engine) compilada
    if engine_ref:
        for search_dir in search_dirs:
            candidate = resolve_path(search_dir) / engine_ref
            if candidate.exists():
                print(f"[RTX 4090 BOOST] Peso em TensorRT detectado e selecionado: {candidate.name}")
                return str(candidate.resolve())

    # Segundo passo de busca: buscar pelo peso padrão (.pt)
    for search_dir in search_dirs:
        candidate = resolve_path(search_dir) / weight_ref
        if candidate.exists():
            return str(candidate.resolve())

    # 3. Busca no diretório atual do notebook
    if engine_ref:
        candidate = resolve_path(engine_ref)
        if candidate.exists():
            print(f"[RTX 4090 BOOST] Peso em TensorRT detectado localmente: {candidate.name}")
            return str(candidate.resolve())

    candidate = resolve_path(weight_ref)
    if candidate.exists():
        return str(candidate.resolve())

    # Retorna a referência padrão para que o Ultralytics gerencie o download automático
    return weight_ref


# Validação física do arquivo de vídeo de entrada
if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Vídeo de entrada não encontrado no caminho configurado: {VIDEO_PATH}\n"
        f"Por favor, verifique se o arquivo está na subpasta correspondente."
    )

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"O OpenCV não conseguiu abrir o arquivo de vídeo: {VIDEO_PATH}")

# Extração de Metadados do Vídeo para dimensionamento e estatísticas
VIDEO_META = {
    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
}
cap.release()

# Cálculos complementares para fins de planejamento de VRAM e performance
VIDEO_META["duration_sec"] = VIDEO_META["frame_count"] / VIDEO_META["fps"] if VIDEO_META["fps"] else 0.0
VIDEO_META["decoded_size_gib"] = (
    VIDEO_META["width"] * VIDEO_META["height"] * 3 * VIDEO_META["frame_count"] / 1024**3
)

# Resolução dinâmica dos pesos
WEIGHTS_REF = resolve_weight_reference(PATH_CONFIG)

print("=" * 60)
print("             METADADOS DO VÍDEO & CONFIGURAÇÃO DO MODELO")
print("=" * 60)
print(json.dumps(VIDEO_META, indent=2))
print("-" * 60)
print(f"Referência de Pesos YOLO: {WEIGHTS_REF}")
print(f"Task de Inferência Ativa: {CONFIG.get('inference', {}).get('task', 'detect')}")
print("=" * 60)
print("Nota: O tamanho descompactado do vídeo completo é exibido para contexto de hardware;")
print("este notebook utiliza streaming por lotes (batched streaming) para poupar RAM do sistema.")

# --- CELL 4 ---
# ============================================================================
# CARREGAMENTO DO MODELO YOLO26 - OTIMIZADO PARA HEAD DETECTION (RTX 4090)
# ============================================================================

INFERENCE_CONFIG = CONFIG.get("inference", {})
expected_task = INFERENCE_CONFIG.get("task", "detect") # Padrão alterado para 'detect'

# Inicializa o YOLO26 com os pesos resolvidos e a tarefa selecionada (detect ou obb)
model = YOLO(WEIGHTS_REF, task=expected_task)

# Validação dinâmica da arquitetura carregada com a configuração desejada no data.yaml
loaded_task = getattr(model, "task", None)
if loaded_task != expected_task:
    raise RuntimeError(
        f"A tarefa do modelo carregado é '{loaded_task}', mas o data.yaml espera '{expected_task}'.\n"
        f"Por favor, verifique se os pesos ('{WEIGHTS_REF}') correspondem à tarefa configurada."
    )

# Transferência ultra rápida do modelo para a memória VRAM dedicada da RTX 4090
if torch.cuda.is_available():
    model.to(f"cuda:{DEVICE}")
    print(f"[RTX 4090] Modelo alocado com sucesso na GPU de ID: {DEVICE}")

# Ajuste dinâmico das classes selecionadas
classes = INFERENCE_CONFIG.get("classes", None)
if classes == []:
    classes = None

# Configuração refinada dos argumentos de inferência para ganho de performance e precisão
PREDICT_ARGS = {
    "device": DEVICE,
    "imgsz": int(INFERENCE_CONFIG.get("imgsz", 1280)),
    "conf": float(INFERENCE_CONFIG.get("conf", 0.25)), # Recomendado 0.25 para cabeças (evita falso positivo)
    "iou": float(INFERENCE_CONFIG.get("iou", 0.50)),   # Reduzido de 0.70 para 0.50 para separar cabeças juntas
    "max_det": int(INFERENCE_CONFIG.get("max_det", 1500)), # Elevado para suportar multidões massivas
    "classes": classes,
    "augment": bool(INFERENCE_CONFIG.get("augment", False)), # Mantido False para maximizar o FPS na 4090
    "half": bool(INFERENCE_CONFIG.get("half", True)),       # Força FP16 nativo na GPU (dobra velocidade)
    "verbose": bool(INFERENCE_CONFIG.get("verbose", False)),
    "stream": False, # Desativado aqui pois o loop do notebook gerencia o batch streaming
}

print("\n[SUCESSO] Modelo carregado e pronto para contagem massiva de cabeças.")
print("Parâmetros ativos de predição:")
print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in PREDICT_ARGS.items()}, indent=2))

# --- CELL 5 ---
# ============================================================================
# CONFIGURAÇÃO DE LÓGICA DE CONTAGEM - FOCO EM CABEÇAS (RTX 4090)
# ============================================================================

COUNTING_CONFIG = CONFIG.get("counting", {})
INFERENCE_CONFIG = CONFIG.get("inference", {})

# Método de contagem: Para cabeças, usamos estritamente "boxes" (caixas delimitadoras)
COUNT_SOURCE = COUNTING_CONFIG.get("count_source", "boxes")

# Como não estamos usando modelo de pose, estas variáveis de pontos-chave são desativadas
# ou mantidas como False para evitar qualquer processamento desnecessário na CPU
REQUIRE_KEYPOINTS = False
MIN_VISIBLE_KEYPOINTS = 0
MIN_KEYPOINT_CONF = 0.0

# Lendo as configurações diretamente do CONFIG para evitar NameError (independente da ordem das células)
CONF_THRESHOLD = float(INFERENCE_CONFIG.get("conf", 0.25))
IMGSZ_RESOLUTION = int(INFERENCE_CONFIG.get("imgsz", 1280))

print("=" * 60)
print("             CONFIGURAÇÕES DE CONTAGEM ATIVAS")
print("=" * 60)
print(f"Método de Contagem:            {COUNT_SOURCE.upper()}")
print(f"Exigir Pontos-Chave (Pose):    {REQUIRE_KEYPOINTS} (Desativado para máxima velocidade)")
print(f"Filtro de Confiança de Caixas: {CONF_THRESHOLD}")
print(f"Resolução de Inferência:       {IMGSZ_RESOLUTION}px")
print("=" * 60)

# Função de validação rápida de detecção de cabeça
def is_valid_detection(box) -> bool:
    """
    Determina se uma caixa de detecção é válida para contagem.
    Como estamos usando detecção de cabeças direta, qualquer caixa que passe
    pelo filtro de confiança inicial do YOLO (conf) é considerada válida.
    """
    # Se você quiser adicionar filtros adicionais (ex: tamanho mínimo de pixels da cabeça),
    # pode fazer isso facilmente aqui. Exemplo:
    # w, h = box.xywh[0][2], box.xywh[0][3]
    # if w < 4 or h < 4: return False
    
    return True

# --- CELL 6 ---
# ============================================================================
# LÓGICA PRINCIPAL DE PREDIÇÃO E PROCESSAMENTO DE VÍDEO - RTX 4090
# ============================================================================

def predict_batch(frames: list[np.ndarray]) -> list[Any]:
    """
    Processa um lote de frames usando a RTX 4090, com proteção de memória.
    """
    if not frames:
        return []

    args = dict(PREDICT_ARGS)
    args["batch"] = len(frames)

    try:
        # torch.inference_mode() é ainda mais rápido e consome menos memória que torch.no_grad()
        with torch.inference_mode():
            return list(model.predict(source=frames, **args))
    except torch.cuda.OutOfMemoryError:
        is_oom = True
    except RuntimeError as error:
        is_oom = "out of memory" in str(error).lower()
        if not is_oom:
            raise

    # Se a memória acabar, a rotina reduz o batch pela metade recursivamente
    if not RUNTIME_CONFIG.get("auto_reduce_batch_on_oom", True) or len(frames) == 1:
        raise RuntimeError("CUDA sem memória (OOM) ao processar um único frame ou redução automática está desativada.")

    torch.cuda.empty_cache()
    midpoint = max(1, len(frames) // 2)
    print(f"\n[AVISO] Reduzindo tamanho do batch para evitar estouro de VRAM na RTX 4090. Novo batch: {midpoint}")
    return predict_batch(frames[:midpoint]) + predict_batch(frames[midpoint:])


def format_seconds(seconds: float) -> str:
    """Formatação utilitária para exibição de ETA."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def result_person_count(result: Any) -> int:
    """
    Lógica de contagem super rápida, focada estritamente em 'boxes' (cabeças).
    Ignora keypoints (pontos-chave), pois estamos usando YOLOv26-Detect.
    """
    if result.boxes is None:
        return 0
    return len(result.boxes)


def annotate_result(result: Any, count: int, frame_idx: int, timestamp: float) -> np.ndarray:
    """
    Desenha as caixas delimitadoras e o contador de cabeças no frame.
    Lógica limpa, ideal para eventos de alta densidade sem poluir a imagem.
    """
    overlay_config = OUTPUT_CONFIG.get("overlay", {})
    
    # Se o overlay estiver desligado, apenas retorna o frame puro
    if not overlay_config.get("enabled", True):
        return result.orig_img.copy()

    # Desenha apenas as caixas, omitindo rótulos (nomes) para não poluir
    annotated = result.plot(
        labels=False, 
        conf=False, 
        boxes=True,
        line_width=overlay_config.get("thickness", 1) # Linhas finas para cabeças pequenas
    )
    
    font_scale = overlay_config.get("font_scale", 0.6)
    thickness = overlay_config.get("thickness", 2)
    
    # Texto principal no canto superior esquerdo
    cv2.putText(
        annotated, 
        f"Cabecas Detectadas: {count}", 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale * 1.5, 
        (0, 0, 255), # Vermelho vibrante
        thickness, 
        cv2.LINE_AA
    )
    
    # Timestamp e FPS no canto inferior esquerdo (ajuda muito em auditorias de segurança)
    cv2.putText(
        annotated, 
        f"Frame: {frame_idx} | Tempo: {timestamp:.2f}s", 
        (20, annotated.shape[0] - 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        (255, 255, 255), 
        max(1, thickness - 1), 
        cv2.LINE_AA
    )
    
    return annotated


def save_snapshot(annotated: np.ndarray, frame_index: int) -> None:
    """Salva um frame em formato JPG na pasta snapshots_dir."""
    filename = SNAPSHOTS_DIR / f"snapshot_frame_{frame_index:06d}.jpg"
    cv2.imwrite(str(filename), annotated)


def build_summary(counts: list[int], elapsed_sec: float, processed_frames: int) -> dict[str, Any]:
    """Cria o objeto JSON com o sumário completo do processamento do vídeo."""
    if counts:
        p95 = float(np.percentile(np.asarray(counts), 95))
        summary_counts = {
            "min_people_in_frame": int(min(counts)),
            "max_people_in_frame": int(max(counts)),
            "mean_people_per_frame": round(float(statistics.fmean(counts)), 3),
            "median_people_per_frame": round(float(statistics.median(counts)), 3),
            "p95_people_per_frame": round(p95, 3),
        }
    else:
        summary_counts = {
            "min_people_in_frame": 0, "max_people_in_frame": 0,
            "mean_people_per_frame": 0.0, "median_people_per_frame": 0.0,
            "p95_people_per_frame": 0.0,
        }

    return {
        "app": APP_CONFIG.get("name"),
        "video": str(VIDEO_PATH),
        "weights": str(WEIGHTS_REF),
        "device": DEVICE,
        "video_meta": VIDEO_META,
        "processed_frames": processed_frames,
        "elapsed_sec": round(float(elapsed_sec), 3),
        "fps_processed": round(processed_frames / elapsed_sec, 3) if elapsed_sec else 0.0,
        "inference": PREDICT_ARGS,
        "counts": summary_counts,
        "outputs": {
            "annotated_video": str(ANNOTATED_VIDEO_PATH),
            "frame_counts_csv": str(FRAME_COUNTS_CSV),
            "summary_json": str(SUMMARY_JSON),
            "snapshots_dir": str(SNAPSHOTS_DIR),
        },
    }


def run_counting() -> dict[str, Any]:
    """Loop principal de leitura, batching, inferência e gravação."""
    batch_size = max(1, int(RUNTIME_CONFIG.get("batch_size", 32))) # Na RTX 4090, o padrão 32 é ideal
    vid_stride = max(1, int(INFERENCE_CONFIG.get("vid_stride", 1)))
    progress_every = max(1, int(OUTPUT_CONFIG.get("progress_every_n_frames", 300)))
    snapshot_every = int(OUTPUT_CONFIG.get("save_snapshot_every_n_frames", 30))
    expected_processed = (VIDEO_META["frame_count"] + vid_stride - 1) // vid_stride

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV não conseguiu abrir o vídeo: {VIDEO_PATH}")

    video_writer = None
    if OUTPUT_CONFIG.get("save_annotated_video", True):
        output_fps = max(1, round(VIDEO_META["fps"])) if VIDEO_META["fps"] else 24
        codec_attempts = ["mp4v", "avc1", "H264", "MJPG", "DIVX", "WMV1"]
        selected_codec = None
        
        for codec in codec_attempts:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                vw = cv2.VideoWriter(
                    str(ANNOTATED_VIDEO_PATH),
                    fourcc,
                    output_fps,
                    (VIDEO_META["width"], VIDEO_META["height"]),
                )
                if vw.isOpened():
                    video_writer = vw
                    selected_codec = codec
                    print(f"✓ Video writer inicializado (Codec: {codec} | FPS: {output_fps})")
                    break
                else:
                    vw.release()
            except Exception:
                continue
        
        if video_writer is None:
            print(f"AVISO: Não foi possível iniciar o vídeo de saída com os codecs padrão.")
            print(f"Pular geração de vídeo. Os arquivos CSV e JSON ainda serão criados normalmente.")

    csv_file = None
    csv_writer = None
    if OUTPUT_CONFIG.get("save_frame_counts", True):
        csv_file = FRAME_COUNTS_CSV.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["frame_index", "timestamp_sec", "people_count"])
        csv_writer.writeheader()

    counts: list[int] = []
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    processed_frames = 0
    frame_index = 0
    batch_number = 0
    next_progress = progress_every
    start_time = time.time()

    def flush_batch() -> None:
        """Envia o batch de frames armazenados para a RTX 4090 e processa os retornos."""
        nonlocal batch_number, processed_frames, next_progress
        if not batch_frames:
            return

        frames = list(batch_frames)
        indices = list(batch_indices)
        batch_frames.clear()
        batch_indices.clear()

        # Envia para a GPU
        results = predict_batch(frames)
        
        if len(results) != len(indices):
            raise RuntimeError(f"Esperados {len(indices)} resultados de predição, recebidos {len(results)}.")

        # Processamento do retorno (Desenhando caixas e computando o CSV)
        for result, source_frame, source_index in zip(results, frames, indices):
            timestamp_sec = source_index / VIDEO_META["fps"] if VIDEO_META["fps"] else 0.0
            
            # Utiliza a função otimizada de contagem baseada apenas em 'boxes'
            count = result_person_count(result)
            counts.append(count)

            if csv_writer is not None:
                csv_writer.writerow({
                    "frame_index": source_index,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "people_count": count,
                })

            if video_writer is not None or (snapshot_every > 0 and source_index % snapshot_every == 0):
                annotated = annotate_result(result, count, source_index, timestamp_sec)
                
                # Previne crash caso a resolução tenha mudado
                if annotated.shape[1] != VIDEO_META["width"] or annotated.shape[0] != VIDEO_META["height"]:
                    annotated = cv2.resize(annotated, (VIDEO_META["width"], VIDEO_META["height"]))

                if video_writer is not None:
                    video_writer.write(annotated)
                if snapshot_every > 0 and source_index % snapshot_every == 0:
                    save_snapshot(annotated, source_index)

            processed_frames += 1

        batch_number += 1
        
        # Gestão inteligente de VRAM (Limpeza de cache da RTX 4090)
        clear_every = int(RUNTIME_CONFIG.get("empty_cuda_cache_every_batches", 0))
        if clear_every > 0 and batch_number % clear_every == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Feedback em tempo real
        while processed_frames >= next_progress:
            elapsed = time.time() - start_time
            rate = processed_frames / elapsed if elapsed else 0.0
            remaining = (expected_processed - processed_frames) / rate if rate else 0.0
            print(
                f"Processados {processed_frames}/{expected_processed} frames "
                f"({rate:0.2f} FPS) | ETA: {format_seconds(remaining)}"
            )
            next_progress += progress_every

    try:
        # Loop de leitura dos frames originais do vídeo
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % vid_stride == 0:
                batch_frames.append(frame)
                batch_indices.append(frame_index)
                
                if len(batch_frames) >= batch_size:
                    flush_batch()

            frame_index += 1

        # Processa frames remanescentes (se a divisão do vídeo não for exata com o tamanho do batch)
        flush_batch()
        
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if csv_file is not None:
            csv_file.close()

    elapsed_sec = time.time() - start_time
    summary = build_summary(counts, elapsed_sec, processed_frames)

    if OUTPUT_CONFIG.get("save_summary", True):
        with SUMMARY_JSON.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

    print("=" * 60)
    print("✓ PROCESSAMENTO FINALIZADO!")
    print(f"Média final: {summary['fps_processed']} FPS")
    print("=" * 60)
    return summary

# Dispara a execução principal e salva na variável do Notebook
SUMMARY = run_counting()
SUMMARY

# --- CELL 7 ---
# ============================================================================
# EXIBIÇÃO DE RESULTADOS E VÍDEO FINAL
# ============================================================================
from IPython.display import Video, display

print("=" * 60)
print("                RESUMO DOS ARQUIVOS GERADOS")
print("=" * 60)
print(f"Sumário JSON (Estatísticas): {SUMMARY_JSON}")
print(f"CSV de Contagem Temporal:    {FRAME_COUNTS_CSV}")
print(f"Diretório de Snapshots:      {SNAPSHOTS_DIR}")
print(f"Vídeo Anotado:               {ANNOTATED_VIDEO_PATH}")
print("=" * 60)

# Exibe um mini-dashboard das estatísticas de multidão se o processamento ocorreu bem
if 'SUMMARY' in locals() and 'counts' in SUMMARY:
    print("\n[INFO] Estatísticas de Multidão (Cabeças):")
    for key, value in SUMMARY['counts'].items():
        # Formatação amigável das chaves do dicionário
        label = key.replace('_', ' ').title()
        print(f" ├─ {label}: {value}")
    print("\n")

if ANNOTATED_VIDEO_PATH.exists():
    print("Visualizando Vídeo Anotado...")
    # O embed=False ajuda na performance do Jupyter para arquivos grandes de 1080p/4k
    display(Video(str(ANNOTATED_VIDEO_PATH), embed=False, width=960))
else:
    print(f"[AVISO] O vídeo não foi encontrado no caminho especificado.")
    print("Verifique se 'save_annotated_video' está como 'true' no seu data.yaml.")
