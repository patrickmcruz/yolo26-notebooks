# Scripts de Benchmark e Comparacao de Metricas

Este diretorio contem utilitarios em Python desenvolvidos para benchmark de performance de execucao e analise estatistica comparativa dos resultados obtidos nos testes de contagem de cabecas.

---

## 1. benchmark_pipeline.py

Este script executa testes de estresse e perfilamento (benchmarks) de throughput no modelo YOLO carregado, simulando diferentes tamanhos de lote (batch size), resolucoes de video de saida e modos de gravacao. 

Seu objetivo e medir o impacto do gargalo de CPU induzido pela codificacao e compressao de video do OpenCV VideoWriter contra o processamento da GPU RTX 4090.

### Como Executar:

Abra um terminal no diretorio raiz do projeto e execute:

```bash
python scripts/benchmarks/benchmark_pipeline.py --model notebooks/testing/test-setup-08-heads-counting-enhanced/yolo26x.pt --video notebooks/testing/test-setup-08-heads-counting-enhanced/input/videos/20260329_34-edited.mp4 --frames 500
```

### Argumentos Suportados:

- `--model`: Caminho para o arquivo de pesos do modelo YOLO (.pt ou .engine).
- `--video`: Caminho para o video .mp4 que sera utilizado como entrada.
- `--frames`: Limite de quadros a serem processados por caso de teste (padrao: 500).
- `--device`: ID da GPU CUDA a ser utilizada (padrao: 0).

### O que e testado:

1. **Standard 1080p Write (Batch 32)**: Execucao convencional em lote de 32 com gravacao de video anotado em 1080p nativo.
2. **Standard 1080p Write (Batch 64)**: Execucao com lote de 64 (saturando melhor a RTX 4090) e gravacao em 1080p nativo.
3. **Optimized 720p Write (Batch 64)**: Execucao em lote de 64 com compressao otimizada (redimensionamento do video anotado para 720p).
4. **Optimized 540p Write (Batch 64)**: Execucao em lote de 64 com compressao de alta velocidade (redimensionamento do video anotado para 540p).
5. **No Video Write (Batch 64)**: Inferencia pura e batching assincrono, com gravacao de video totalmente desativada (limite fisico maximo da GPU).

---

## 2. compare_metrics.py

Este script carrega dados de contagem temporal gerados por dois experimentos (arquivos `frame_counts.csv` e `summary.json`) e gera um relatorio estruturado comparando a precisao estatistica e a estabilidade temporal (ruido e flickering) entre as abordagens.

### Como Executar:

```bash
python scripts/benchmarks/compare_metrics.py --dir1 notebooks/testing/test-setup-06-frames --dir2 notebooks/testing/test-setup-08-heads-counting-enhanced --name1 "Setup 06 (Pose)" --name2 "Setup 08 (Heads)"
```

### Argumentos Suportados:

- `--dir1`: Caminho da pasta de saida do Experimento 1 (deve conter a subpasta `output/` com `frame_counts.csv` e `summary.json`).
- `--dir2`: Caminho da pasta de saida do Experimento 2 (deve conter a subpasta `output/` com `frame_counts.csv` e `summary.json`).
- `--name1`: Nome curto para identificar o Experimento 1 na tabela.
- `--name2`: Nome curto para identificar o Experimento 2 na tabela.

### Metricas Analisadas na Comparacao:

- **Frames Processados**: Numero total de quadros processados no experimento.
- **Tempo de Execucao & Throughput**: Tempo decorrido em segundos e media global de FPS.
- **Media & Desvio Padrao**: Media de objetos por quadro e a dispersao da contagem.
- **Ruido Temporal Medio (Frame-a-Frame)**: A diferenca absoluta media de contagem de um frame para o outro. Mede a instabilidade e o "flickering" do modelo.
- **Taxa de Instabilidade Severa**: Percentual de frames onde a contagem oscilou mais do que 3 unidades de forma instantanea entre quadros consecutivos.
