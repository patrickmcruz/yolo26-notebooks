import cv2
import torch
import time
import sys
import queue
import threading
import argparse
from pathlib import Path
from ultralytics import YOLO

# Configura stdout para UTF-8 para evitar problemas de codificacao no Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 Performance Pipeline Benchmark Tool")
    parser.add_argument(
        "--model",
        type=str,
        default=r"notebooks/testing/test-setup-08-heads-counting-enhanced/yolo26x.pt",
        help="Caminho para os pesos do modelo YOLO (.pt ou .engine)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=r"notebooks/testing/test-setup-08-heads-counting-enhanced/input/videos/20260329_34-edited.mp4",
        help="Caminho para o video de entrada (.mp4)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=500,
        help="Numero de frames para processar em cada teste do benchmark"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID do dispositivo CUDA (GPU)"
    )
    return parser.parse_args()

def run_benchmark(name, model, video_path, device_id, num_frames, batch_size, write_video, output_res=None):
    # Dimensoes do video original
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # Inicializa as filas
    f_queue = queue.Queue(maxsize=128)
    w_queue = queue.Queue(maxsize=128)
    stop_read = False
    
    # Thread do leitor
    def reader_worker():
        reader_cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        while reader_cap.isOpened() and not stop_read and frame_idx < num_frames:
            ok, frame = reader_cap.read()
            if not ok:
                break
            f_queue.put((frame, frame_idx), block=True)
            frame_idx += 1
        reader_cap.release()
        f_queue.put((None, None))

    # Thread do gravador
    def writer_worker(video_writer_ref):
        while True:
            item = w_queue.get(block=True)
            if item is None:
                break
            result, source_index = item
            if write_video:
                annotated = result.plot(labels=False, conf=False, boxes=True, line_width=1)
                if output_res:
                    annotated = cv2.resize(annotated, output_res)
                if video_writer_ref is not None:
                    video_writer_ref.write(annotated)

    vw = None
    if write_video:
        target_res = output_res if output_res else (width, height)
        vw = cv2.VideoWriter(
            "temp_benchmark.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            int(fps) if fps > 0 else 30,
            target_res
        )

    # Inicia as threads
    r_thread = threading.Thread(target=reader_worker, daemon=True)
    w_thread = threading.Thread(target=writer_worker, args=(vw,), daemon=True)
    
    r_thread.start()
    w_thread.start()
    
    # Loop principal
    batch_frames = []
    batch_indices = []
    processed_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame, idx = f_queue.get(block=True)
            if frame is None:
                break
            batch_frames.append(frame)
            batch_indices.append(idx)
            
            if len(batch_frames) >= batch_size:
                args = {
                    "device": device_id, 
                    "imgsz": 1280, 
                    "half": True, 
                    "conf": 0.25, 
                    "iou": 0.50, 
                    "max_det": 1500, 
                    "verbose": False, 
                    "batch": len(batch_frames)
                }
                with torch.inference_mode():
                    results = list(model.predict(source=batch_frames, **args))
                
                for r, idx_s in zip(results, batch_indices):
                    if write_video:
                        w_queue.put((r, idx_s), block=True)
                    processed_count += 1
                
                batch_frames.clear()
                batch_indices.clear()
        
        # Processa frames remanescentes
        if batch_frames:
            args = {
                "device": device_id, 
                "imgsz": 1280, 
                "half": True, 
                "conf": 0.25, 
                "iou": 0.50, 
                "max_det": 1500, 
                "verbose": False, 
                "batch": len(batch_frames)
            }
            with torch.inference_mode():
                results = list(model.predict(source=batch_frames, **args))
            for r, idx_s in zip(results, batch_indices):
                if write_video:
                    w_queue.put((r, idx_s), block=True)
                processed_count += 1
    finally:
        stop_read = True
        w_queue.put(None)
        
        r_thread.join(timeout=2.0)
        w_thread.join(timeout=5.0)
        
        if vw is not None:
            vw.release()
            try:
                Path("temp_benchmark.mp4").unlink(missing_ok=True)
            except Exception:
                pass
            
    elapsed = time.time() - start_time
    fps_result = processed_count / elapsed if elapsed else 0.0
    return elapsed, fps_result, processed_count

def main():
    args = parse_args()
    model_path = Path(args.model)
    video_path = Path(args.video)

    if not model_path.exists():
        print(f"Erro: Modelo nao encontrado em {model_path}")
        sys.exit(1)
    if not video_path.exists():
        print(f"Erro: Video nao encontrado em {video_path}")
        sys.exit(1)

    print("Inicializando modelo YOLO...")
    model = YOLO(str(model_path))
    if torch.cuda.is_available():
        model.to(f"cuda:{args.device}")
        
    print("Modelo carregado com sucesso.")
    print(f"Executando benchmarks (limite de {args.frames} frames por teste)...")

    benchmarks = [
        ("Standard 1080p Write (Batch 32)", 32, True, None),
        ("Standard 1080p Write (Batch 64)", 64, True, None),
        ("Optimized 720p Write (Batch 64)", 64, True, (1280, 720)),
        ("Optimized 540p Write (Batch 64)", 64, True, (960, 540)),
        ("No Video Write - Pure Inference (Batch 64)", 64, False, None),
    ]

    results = []
    for name, batch_size, write_video, res in benchmarks:
        try:
            elapsed, fps_val, processed = run_benchmark(
                name, model, video_path, args.device, args.frames, batch_size, write_video, res
            )
            results.append((name, batch_size, str(res) if res else "Original", "Ativado" if write_video else "Desativado", f"{elapsed:.2f}s", f"{fps_val:.2f} FPS"))
        except Exception as e:
            results.append((name, batch_size, "Erro", "Erro", "Erro", str(e)))

    print("\n" + "="*80)
    print(f"{'RESULTADOS DO BENCHMARK DE PERFORMANCE':^80}")
    print("="*80)
    print(f"{'Caso de Teste':<40} | {'Batch':<5} | {'Resolucao':<10} | {'Gravacao':<8} | {'Tempo':<7} | {'FPS':<10}")
    print("-"*80)
    for name, batch, res, write, t, fps_str in results:
        print(f"{name:<40} | {batch:<5} | {res:<10} | {write:<8} | {t:<7} | {fps_str:<10}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
