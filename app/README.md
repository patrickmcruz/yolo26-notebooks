# YOLO26 Head Counting - Aplicação Isolada (/app)

Esta pasta contém o subprojeto **`app`**, que implementa de forma 100% isolada e autônoma o pipeline de detecção e contagem de cabeças humanas. 

Seguindo as novas especificações, esta aplicação **não possui dependências de arquivos ou entradas externas**, contendo todos os recursos necessários (vídeos, pesos de modelos, requerimentos e testes) dentro da pasta `/app`.

---

## 📚 Referências e Créditos de Modelos

Os pesos utilizados neste subprojeto (`best.pt` / `best.engine`) baseiam-se em modelos pré-treinados e disponibilizados publicamente pela comunidade de código aberto:

* **Modelo Original**: [irail-crowd-counting-yolov8n](https://huggingface.co/AmineSam/irail-crowd-counting-yolov8n)
* **Autor/Mantenedor**: [AmineSam (HuggingFace)](https://huggingface.co/AmineSam)
* **Descrição**: Um modelo especializado baseado na arquitetura Ultralytics YOLOv8n, otimizado para detecção densa de cabeças e contagem de pessoas em cenários de alta aglomeração (como plataformas de embarque e áreas públicas).

---

## 🏗️ Estrutura da Aplicação Isolada

```
count-github-yolo-01/app/
├── head_counting/                     # Pacote Python contendo a lógica central
│   ├── __init__.py                    # Interface de exports da biblioteca
│   ├── config.py                      # Parser dataclass tipado e resolução de caminhos locais
│   ├── model.py                       # Setup CUDA, compilação de pesos YOLO e inferência batch
│   ├── video.py                       # Threads de leitura e anotação/gravação de vídeo
│   └── pipeline.py                    # Orquestração de ponta a ponta do processo
│
├── input/
│   └── videos/                        # Vídeos originais para processamento
│       ├── muralha-dia.mp4           # Vídeo Diurno
│       └── muralha-noite.mp4          # Vídeo Noturno
│
├── weights/
│   └── best.pt                        # Pesos do modelo pré-treinado YOLO (especializado em cabeças)
│
├── tests/                             # Suíte de testes unitários
│   ├── test_config.py                 # Valida o carregamento e resolução do config
│   └── test_pipeline.py               # Mocka o processamento para validar o loop principal
│
├── data_day.yaml                      # Configuração para o cenário Diurno
├── data_night.yaml                    # Configuração para o cenário Noturno
├── run.py                             # CLI Runner executável
├── test-setup-11-heads-counting-enhanced.ipynb # Jupyter Notebook para execução visual e relatórios
├── requirements.txt                   # Arquivo de requerimentos isolado para a aplicação
└── README.md                          # Este manual de instrução
```

---

## 🛠️ Modificações e Padrões Implementados

- **Isolamento de Entradas**: Os arquivos de vídeo de entrada e os pesos dos modelos foram movidos para dentro de `/app/input/videos/` e `/app/weights/`, respectivamente.
- **Configurações Locais**: Os arquivos `data_day.yaml` e `data_night.yaml` foram atualizados para buscar os caminhos internos (`input/videos/...` e `weights/best.pt`).
- **Resolução de Caminhos Absolutos**: O módulo de configuração resolve todos os caminhos de forma absoluta em tempo de execução com base no diretório em que o arquivo `.yaml` correspondente se encontra.
- **Requerimentos Autônomos**: Um arquivo `requirements.txt` próprio foi adicionado à pasta `/app` com todas as dependências isoladas.

---

## 🚀 Como Executar

Abra o terminal na pasta `/app` do projeto:

```bash
cd app
```

### Executar via Terminal (CLI)
Você pode rodar o processador de vídeo direto no console para o cenário desejado:

* **Cenário Diurno**:
  ```bash
  python run.py --config data_day.yaml
  ```

* **Cenário Noturno**:
  ```bash
  python run.py --config data_night.yaml
  ```

Os resultados (estatísticas em JSON, contagem frame-a-frame em CSV e vídeos anotados) serão gerados respectivamente em `output/day/` e `output/night/`.

### Executar via Jupyter Notebook
Abra o arquivo [`test-setup-11-heads-counting-enhanced.ipynb`](count-github-yolo-01/app/test-setup-11-heads-counting-enhanced.ipynb) no VSCode. Escolha o arquivo de configuração na primeira célula (`data_day.yaml` ou `data_night.yaml`) e execute as células sequencialmente.

---

## 🧪 Rodar os Testes Unitários

Para garantir o funcionamento completo das peças isoladas (sem depender de hardware de GPU CUDA ativo), execute os testes unitários integrados:

```bash
python -m pytest tests/
```
