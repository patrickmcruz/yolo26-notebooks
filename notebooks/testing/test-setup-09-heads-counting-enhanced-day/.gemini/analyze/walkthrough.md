# Walkthrough de Alteracoes: Deteccao de Cabecas com Modelo do Hugging Face (Setup 09)

Este documento resume as modificacoes efetuadas para implementar a **Abordagem B (Simplificada)** de deteccao de cabecas no notebook do Setup 09.

---

## Modificacoes Efetuadas

### 1. Configuracao Geral

#### [data.yaml](count-github-yolo-01/notebooks/testing/test-setup-09-heads-counting-enhanced/data.yaml)
* O parametro `app.name` foi atualizado de `test-setup-08-heads-counting-enhanced` para `test-setup-09-heads-counting-enhanced`.
* O parametro `paths.weights` foi alterado para `"best.pt"`.
* **Correção de Pesos do Hugging Face**: Para evitar falhas de `FileNotFoundError` na biblioteca `ultralytics` (que não realiza o download automático em algumas versões), baixamos o arquivo `best.pt` programaticamente a partir da URL oficial do repositório `AmineSam/irail-crowd-counting-yolov8n` no Hugging Face e configuramos o `data.yaml` para utilizá-lo de forma local e offline.
* O nome do video de saida anotado foi modificado para `20260329_34-edited_yolo26_head_count_crowdhuman.mp4`.

---

### 2. Jupyter Notebook

#### [test-setup-09-heads-couting-enhanced.ipynb](count-github-yolo-01/notebooks/testing/test-setup-09-heads-counting-enhanced/test-setup-09-heads-couting-enhanced.ipynb)
* **Celula 1**: Corrigido o erro de busca de pasta (`target_subdir` e `alt_subdir`), mudando de `test-setup-09-heads-couting-enhanced` (sem 'n') para `test-setup-09-heads-counting-enhanced` (com 'n'), garantindo que a funcao `find_notebook_dir()` localize com sucesso o `data.yaml` correto deste setup.
* **Celula 6**: Substituida inteiramente pelo codigo de inferencia de alta performance otimizado para a RTX 4090:
  * Adicionado suporte ao redimensionamento inteligente (`output_resolution` do `data.yaml`) para aliviar a carga de compressao na CPU.
  * Adicionado suporte a pipeline assincrona real de escrita (`video_writer_worker` rodando em segundo plano), desacoplando a inferencia da GPU do gargalo de gravação do OpenCV.
* **Estrutura e Documentação**: Adicionadas novas células Markdown detalhadas acima de cada célula de código (de 1 a 7) e ajustado o cabeçalho introdutório do notebook (Célula 0) em português, esclarecendo o fluxo de execução completo do pipeline para o usuário antes da execução de cada etapa.


---

## Como Executar e Validar

1. Abra o arquivo [test-setup-09-heads-couting-enhanced.ipynb](count-github-yolo-01/notebooks/testing/test-setup-09-heads-counting-enhanced/test-setup-09-heads-couting-enhanced.ipynb) no VS Code ou Jupyter.
2. Certifique-se de que o seu ambiente virtual Python esteja ativo e que possua conexao com a Internet (apenas para a primeira execucao, onde os pesos serao baixados).
3. Execute todas as celulas em sequencia.
4. Na **Celula 4**, voce devera ver o log do `ultralytics` baixando e salvando em cache os pesos do Hugging Face.
5. Na **Celula final**, assista ao video anotado gerado para verificar a precisao das bounding boxes limitadas apenas as cabecas.
