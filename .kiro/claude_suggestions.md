Preciso desenvolver um projeto COMPLETO de Análise de Sentimentos em 
Avaliações de Produtos para disciplina de Processamento Natural de Linguagem, 
combinando SIMPLICIDADE DE EXECUÇÃO com RIGOR CIENTÍFICO.

OBJETIVO: Comparar três abordagens clássicas + uma moderna (bônus) em 
reviews de produtos, com validação estatística robusta e análises avançadas 
específicas de NLP.

═══════════════════════════════════════════════════════════════════════════
PARTE 1: CONFIGURAÇÃO BÁSICA
═══════════════════════════════════════════════════════════════════════════

1. COLETA DE DADOS:
   - Produto selecionado: [escolher um produto específico]
   - Fonte: Kaggle, Hugging Face, ou web scraping
   - Quantidade MÍNIMA: 3.000 reviews
   - Quantidade IDEAL para validação estatística: 5.000+ reviews
   
   - Conversão de notas para sentimento:
     * Positivo: 4-5 estrelas
     * Negativo: 1-2 estrelas
     * Descartar: 3 estrelas (neutro)
   
   - Divisão: 70% treino, 15% validação, 15% teste
   - Balanceamento: 50% positivo, 50% negativo
   
   - Metadados adicionais a coletar:
     * review_length (número de caracteres)
     * has_emojis (boolean)
     * language_formality (formal/informal - detectar gírias)
     * contains_typos (simular ou detectar)

═══════════════════════════════════════════════════════════════════════════
PARTE 2: MODELOS (4 OBRIGATÓRIOS)
═══════════════════════════════════════════════════════════════════════════

2. IMPLEMENTAR QUATRO MODELOS:

   ┌─────────────────────────────────────────────────────────────────────┐
   │ MODELO 1: SVM + Bag of Words (BoW)                                  │
   ├─────────────────────────────────────────────────────────────────────┤
   │ • Vetorização: TfidfVectorizer (max_features=5000, ngram_range=1-2)│
   │ • Classificador: SVM (kernel='linear', C=1.0)                      │
   │ • Baseline clássico e interpretável                                 │
   └─────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────┐
   │ MODELO 2: SVM + Word Embeddings                                     │
   ├─────────────────────────────────────────────────────────────────────┤
   │ • Embeddings: Word2Vec (Google News 300d) ou GloVe                 │
   │ • Agregação: Média ponderada dos vetores (TF-IDF weights)          │
   │ • Classificador: SVM (kernel='rbf', C=1.0, gamma='scale')          │
   └─────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────┐
   │ MODELO 3: BERT Fine-tuned                                           │
   ├─────────────────────────────────────────────────────────────────────┤
   │ • Modelo: distilbert-base-uncased (mais rápido que BERT completo)  │
   │ • Fine-tuning: 3-5 épocas, batch_size=16, lr=2e-5                  │
   │ • Early stopping com validação                                      │
   └─────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────┐
   │ MODELO 4: LLM com In-Context Learning (BÔNUS)                      │
   ├─────────────────────────────────────────────────────────────────────┤
   │ • Escolher: GPT-4, Claude 3.5 Sonnet, ou Gemini Pro                │
   │ • Few-shot: 5 exemplos estratégicos no prompt                      │
   │ • Zero-shot: Para comparação (se tempo permitir)                   │
   └─────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
PARTE 3: VALIDAÇÃO ESTATÍSTICA (RIGOR CIENTÍFICO)
═══════════════════════════════════════════════════════════════════════════

3. MÚLTIPLAS EXECUÇÕES COM DIFERENTES SEEDS:

   a) Número de simulações:
      - MÍNIMO: 10 simulações por modelo (aceitável)
      - IDEAL: 30 simulações por modelo (robusto)
      - Variar: seed para split dos dados + seed de inicialização (BERT)
   
   b) Para cada simulação extrair:
      ✓ Accuracy
      ✓ Precision (macro e por classe)
      ✓ Recall (macro e por classe)
      ✓ F1-Score (macro e weighted)
      ✓ Tempo de treinamento (se aplicável)
      ✓ Tempo de inferência (média por review)
   
   c) Armazenar:
      - CSV com métricas de TODAS as simulações
      - Formato: [modelo, simulacao_id, accuracy, precision, recall, f1, ...]

4. TESTES ESTATÍSTICOS (α=0.05, 95% confiança):

   ┌─────────────────────────────────────────────────────────────────────┐
   │ TESTE 1: Wilcoxon Signed-Rank Test (pareado)                       │
   ├─────────────────────────────────────────────────────────────────────┤
   │ • Objetivo: Comparar pares de modelos (ex: BERT vs SVM+BoW)        │
   │ • Hipótese: H0 = não há diferença significativa                    │
   │ • Executar: Para cada par de modelos, em cada métrica              │
   │ • Reportar: p-valor, significância (p<0.05?), vencedor             │
   │                                                                      │
   │ Exemplo de comparações:                                             │
   │ - BERT vs SVM+BoW                                                   │
   │ - BERT vs SVM+Embeddings                                            │
   │ - BERT vs LLM                                                       │
   │ - SVM+Embeddings vs SVM+BoW                                         │
   │ - LLM vs todos (se tempo permitir)                                 │
   └─────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────┐
   │ TESTE 2: Kruskal-Wallis H-test (não-paramétrico, múltiplos grupos)│
   ├─────────────────────────────────────────────────────────────────────┤
   │ • Objetivo: Testar se HÁ diferença entre os 4 modelos              │
   │ • Hipótese: H0 = todos os modelos têm mesma mediana                │
   │ • Executar: Uma vez para cada métrica (Accuracy, F1, etc.)         │
   │ • Se p<0.05: Há diferença significativa → prosseguir com Wilcoxon  │
   └─────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────┐
   │ TESTE 3: Teste de Normalidade (Shapiro-Wilk)                       │
   ├─────────────────────────────────────────────────────────────────────┤
   │ • Objetivo: Verificar se distribuições são normais                 │
   │ • Se normais: Poderia usar t-test (mais poderoso)                  │
   │ • Se não-normais: Wilcoxon é mais apropriado (esperado em ML)      │
   └─────────────────────────────────────────────────────────────────────┘

5. VISUALIZAÇÕES ESTATÍSTICAS:

   a) Boxplots:
      - Um boxplot POR MÉTRICA (Accuracy, Precision, Recall, F1)
      - Comparando os 4 modelos
      - Mostrar: mediana, quartis, outliers
   
   b) Gráficos de linha:
      - Evolução de Accuracy ao longo das simulações
      - Evolução de F1-Score ao longo das simulações
      - Identificar estabilidade vs variabilidade
   
   c) Tabela de significância:
      - Matriz de p-valores (modelo A vs modelo B)
      - Código de cores: verde (p<0.05), vermelho (p≥0.05)
   
   d) Gráfico de barras com intervalo de confiança:
      - Média ± desvio padrão para cada modelo
      - Ou: Média com intervalo de confiança 95%

═══════════════════════════════════════════════════════════════════════════
PARTE 4: ANÁLISES ADICIONAIS ESPECÍFICAS DE NLP
═══════════════════════════════════════════════════════════════════════════

6. ANÁLISE 1 - EXEMPLOS DE ACERTOS E ERROS:

   Para CADA modelo, identificar e documentar:
   
   a) Casos de ACERTO por todos os modelos:
      - 5-10 exemplos de reviews fáceis
      - Características comuns (claros, diretos, vocabulário simples)
   
   b) Casos onde APENAS UM modelo acerta:
      - BERT acerta, SVMs erram → captura contexto complexo
      - SVM+Embeddings acerta, BoW erra → semântica ajuda
      - LLM acerta, todos erram → generalização superior
   
   c) Casos de ERRO por todos os modelos:
      - Reviews com sarcasmo: "Ótimo! Quebrou no primeiro dia 🙄"
      - Reviews com ironia: "Adorei esperar 3 meses pela entrega"
      - Reviews ambíguos: "É bom... mas esperava mais"
   
   d) Análise de Falsos Positivos e Falsos Negativos:
      - Quais tipos de erro cada modelo comete mais?
      - Padrões linguísticos que confundem cada modelo

   FORMATO DE APRESENTAÇÃO:
   ┌─────────────────────────────────────────────────────────────────────┐
   │ Review: "Produto excelente mas entrega péssima"                     │
   │ Label verdadeiro: NEGATIVO                                          │
   │ ─────────────────────────────────────────────────────────────────── │
   │ SVM+BoW:        POSITIVO ✗ (focou em "excelente")                  │
   │ SVM+Embeddings: NEGATIVO ✓ (capturou "péssima" com peso)           │
   │ BERT:           NEGATIVO ✓ (entendeu contexto "mas")               │
   │ LLM:            NEGATIVO ✓ (raciocínio: entrega > produto)         │
   │ ─────────────────────────────────────────────────────────────────── │
   │ INSIGHT: "mas" é crucial - BoW não captura, outros sim             │
   └─────────────────────────────────────────────────────────────────────┘

7. ANÁLISE 2 - COMPRIMENTO DE TEXTO vs ACURÁCIA:

   a) Binning de comprimento:
      - Curto: 0-50 caracteres
      - Médio: 51-200 caracteres
      - Longo: 201-500 caracteres
      - Muito longo: 500+ caracteres
   
   b) Para cada bin e cada modelo, calcular:
      - Accuracy no bin
      - F1-Score no bin
      - Número de amostras no bin
   
   c) Visualização:
      - Gráfico de linha: Eixo X = comprimento, Eixo Y = Accuracy
      - Uma linha por modelo
   
   d) Análise esperada:
      - BoW: pode degradar em textos muito longos (esparsidade)
      - Embeddings: mais robusto a comprimento
      - BERT: limite de 512 tokens - truncamento afeta?
      - LLM: contexto grande, deve ser robusto
   
   e) Teste estatístico:
      - Correlação de Pearson/Spearman: comprimento × accuracy
      - Por modelo: há correlação significativa?

8. ANÁLISE 3 - ROBUSTEZ A ERROS DE DIGITAÇÃO/GRAMÁTICA:

   a) Criar dataset perturbado:
      - Pegar 200-500 reviews do conjunto de teste
      - Aplicar perturbações controladas:
        * Trocar 5% das letras aleatoriamente (typos)
        * Remover acentos ("ótimo" → "otimo")
        * Duplicar letras ("bom" → "boom")
        * Inverter letras adjacentes ("produto" → "porduto")
   
   b) Avaliar cada modelo:
      - Accuracy no dataset limpo
      - Accuracy no dataset perturbado
      - Queda de performance (Δ Accuracy)
   
   c) Visualização:
      - Gráfico de barras: [Limpo | Perturbado] por modelo
      - Calcular % de degradação
   
   d) Análise esperada:
      - BoW: muito sensível (palavras fora do vocabulário)
      - Embeddings: mais robusto (palavras similares têm vetores próximos)
      - BERT: robusto (subword tokenization - WordPiece)
      - LLM: muito robusto (treinado em textos ruidosos da internet)

9. ANÁLISE 4 - TEXTOS COM EMOJIS:

   a) Separar subconjunto:
      - Reviews COM emojis vs SEM emojis
      - Garantir balanceamento de sentimento em ambos
   
   b) Avaliar cada modelo:
      - Accuracy em reviews SEM emojis
      - Accuracy em reviews COM emojis
      - Diferença estatística (teste t ou Wilcoxon)
   
   c) Análise de emojis informativos:
      - Positivos: 😊 👍 ❤️ ⭐ 🎉
      - Negativos: 😡 👎 💔 😞 ⚠️
      - Neutros: 🤔 😐
   
   d) Teste adicional:
      - Remover emojis dos reviews e reclassificar
      - Emojis são cruciais ou apenas decorativos?
   
   e) Visualização:
      - Gráfico de barras: Accuracy [Com Emojis | Sem Emojis] por modelo
   
   f) Análise esperada:
      - BoW: ignora emojis (trata como tokens desconhecidos)
      - BERT: pode capturar (se fine-tuned com emojis)
      - LLM: forte com emojis (treinamento em redes sociais)

10. ANÁLISE 5 - SARCASMO E IRONIA:

    a) Anotar manualmente subset de sarcasmo:
       - Identificar 50-100 reviews sarcásticos/irônicos no teste
       - Exemplos:
         * "Adorei! Durou impressionantes 2 dias 👏"
         * "Excelente qualidade... se você gosta de plástico barato"
         * "Recomendo se você quer jogar dinheiro fora"
    
    b) Avaliar cada modelo:
       - Accuracy no subset sarcástico
       - Accuracy no subset NÃO-sarcástico
       - Comparar com performance geral
    
    c) Análise de features de sarcasmo:
       - Presença de "!" múltiplos
       - Palavras extremas ("adorei", "excelente") + sentimento negativo
       - Emojis irônicos (👏 🎉 usado negativamente)
    
    d) Visualização:
       - Tabela: Accuracy [Geral | Sarcástico] por modelo
       - % de queda em reviews sarcásticos
    
    e) Análise esperada:
       - BoW: péssimo (só vê palavras positivas, ignora contexto)
       - Embeddings: ligeiramente melhor
       - BERT: melhor (captura contexto, mas ainda desafiador)
       - LLM: melhor performance (raciocínio de alto nível)

11. ANÁLISE 6 - SENSIBILIDADE A IDIOMA/DIALETO/FORMALIDADE:

    a) Categorizar reviews por formalidade:
       - Formal: "O produto apresenta excelente qualidade"
       - Informal: "Produto top demais, curti muito"
       - Gírias: "Produto massa, show de bola, recomendo"
    
    b) Detectar automaticamente:
       - Usar heurísticas simples:
         * Gírias brasileiras: "top", "massa", "show", "da hora"
         * Abreviações: "vc", "tbm", "mt", "blz"
         * ALL CAPS: "ADOREI", "PÉSSIMO"
    
    c) Avaliar cada modelo:
       - Accuracy em cada categoria de formalidade
       - Teste estatístico: diferença significativa entre categorias?
    
    d) Análise adicional - tratamento de caso:
       - Testar em reviews em UPPERCASE
       - Testar em reviews em lowercase
       - Testar em MiXeD CaSe
    
    e) Visualização:
       - Heatmap: Linhas=modelos, Colunas=categorias, Valores=Accuracy
    
    f) Análise esperada:
       - BoW: sensível se treinou lowercase (gírias fora vocabulário)
       - BERT: robusto (case-insensitive por padrão)
       - LLM: muito robusto (viu diversidade linguística enorme)

═══════════════════════════════════════════════════════════════════════════
PARTE 5: ESTRUTURA DE IMPLEMENTAÇÃO
═══════════════════════════════════════════════════════════════════════════

12. ORGANIZAÇÃO DO CÓDIGO:

    projeto_sentiment_nlp/
    ├── data/
    │   ├── raw/                          # Dados originais
    │   ├── processed/                    # Dados limpos
    │   ├── perturbed/                    # Dataset com typos
    │   ├── emoji_analysis/               # Subsets com/sem emojis
    │   ├── sarcasm_subset/               # Reviews sarcásticos anotados
    │   └── formality_categories/         # Formal/Informal/Gírias
    │
    ├── notebooks/
    │   ├── 01_data_collection_eda.ipynb
    │   ├── 02_data_preparation.ipynb
    │   ├── 03_model_svm_bow.ipynb
    │   ├── 04_model_svm_embeddings.ipynb
    │   ├── 05_model_bert.ipynb
    │   ├── 06_model_llm.ipynb
    │   ├── 07_statistical_validation.ipynb
    │   ├── 08_error_analysis.ipynb
    │   ├── 09_length_analysis.ipynb
    │   ├── 10_robustness_typos.ipynb
    │   ├── 11_emoji_analysis.ipynb
    │   ├── 12_sarcasm_analysis.ipynb
    │   └── 13_formality_analysis.ipynb
    │
    ├── src/
    │   ├── data_preprocessing.py
    │   ├── data_perturbation.py          # Adicionar typos, etc.
    │   ├── model_svm_bow.py
    │   ├── model_svm_embeddings.py
    │   ├── model_bert.py
    │   ├── model_llm.py
    │   ├── evaluation_metrics.py
    │   ├── statistical_tests.py          # Wilcoxon, Kruskal-Wallis
    │   ├── error_analysis.py
    │   ├── advanced_analysis.py          # Comprimento, emojis, etc.
    │   └── utils.py
    │
    ├── results/
    │   ├── simulations/                  # 10-30 simulações por modelo
    │   │   ├── svm_bow_simulations.csv
    │   │   ├── svm_emb_simulations.csv
    │   │   ├── bert_simulations.csv
    │   │   └── llm_simulations.csv
    │   ├── statistical_tests/
    │   │   ├── wilcoxon_results.json
    │   │   ├── kruskal_wallis_results.json
    │   │   └── statistical_report.txt
    │   ├── error_analysis/
    │   │   ├── examples_correct_all.txt
    │   │   ├── examples_bert_only.txt
    │   │   ├── examples_incorrect_all.txt
    │   │   └── confusion_matrices/
    │   ├── advanced_analysis/
    │   │   ├── length_vs_accuracy.csv
    │   │   ├── typos_robustness.csv
    │   │   ├── emoji_analysis.csv
    │   │   ├── sarcasm_performance.csv
    │   │   └── formality_analysis.csv
    │   └── plots/
    │       ├── boxplots/
    │       ├── line_plots/
    │       ├── statistical/
    │       └── advanced_analysis/
    │
    ├── presentation/
    │   ├── slides.pptx
    │   ├── video_script.md
    │   └── supplementary_material.pdf
    │
    ├── config.py
    ├── requirements.txt
    └── README.md

13. BIBLIOTECAS NECESSÁRIAS:

    # Básicas
    pandas>=1.5.0
    numpy>=1.23.0
    matplotlib>=3.6.0
    seaborn>=0.12.0
    
    # NLP Clássico
    scikit-learn>=1.2.0
    nltk>=3.8
    gensim>=4.3.0
    
    # BERT
    transformers>=4.35.0
    torch>=2.0.0
    datasets>=2.14.0
    accelerate>=0.24.0
    
    # LLM APIs (escolher 1)
    openai>=1.3.0
    anthropic>=0.7.0
    google-generativeai>=0.3.0
    
    # Estatística
    scipy>=1.10.0
    statsmodels>=0.14.0
    
    # Perturbação de texto
    nlpaug>=1.1.11
    
    # Detecção de emoji
    emoji>=2.8.0
    
    # Utilidades
    tqdm>=4.66.0
    joblib>=1.3.0

═══════════════════════════════════════════════════════════════════════════
PARTE 6: APRESENTAÇÃO (15 MINUTOS)
═══════════════════════════════════════════════════════════════════════════

14. ESTRUTURA DOS SLIDES (18-20 slides):

    SLIDE 1:  Título + Objetivo + Motivação
    SLIDE 2:  Dataset (produto, quantidade, balanceamento)
    SLIDE 3:  Metodologia - Visão Geral (4 modelos + validação)
    
    SLIDE 4:  Modelo 1 - SVM + BoW
    SLIDE 5:  Modelo 2 - SVM + Embeddings
    SLIDE 6:  Modelo 3 - BERT Fine-tuned
    SLIDE 7:  Modelo 4 - LLM In-Context (bônus)
    
    SLIDE 8:  Resultados Principais - Tabela Comparativa
    SLIDE 9:  Validação Estatística - Wilcoxon + p-valores
    SLIDE 10: Boxplots - Distribuição das Métricas
    SLIDE 11: Gráficos de Linha - Estabilidade
    
    SLIDE 12: Análise de Erros - Exemplos Qualitativos
    SLIDE 13: Comprimento de Texto vs Acurácia
    SLIDE 14: Robustez a Typos - Queda de Performance
    SLIDE 15: Emojis e Sarcasmo - Desafios Especiais
    SLIDE 16: Sensibilidade a Formalidade/Dialeto
    
    SLIDE 17: Trade-offs - Performance vs Complexidade vs Custo
    SLIDE 18: Conclusões + Recomendações Práticas
    SLIDE 19: Contribuições + Trabalhos Futuros
    SLIDE 20: Agradecimentos + Q&A

15. ROTEIRO DO VÍDEO (15 minutos):

    00:00-01:00  Introdução + Motivação + Dataset
    01:00-04:00  4 Modelos (45 seg cada)
    04:00-06:00  Resultados + Validação Estatística
    06:00-08:00  Análise de Erros Qualitativos
    08:00-11:00  Análises Avançadas (comprimento, typos, emojis, sarcasmo)
    11:00-13:00  Trade-offs + Recomendações
    13:00-14:30  Conclusões + Contribuições
    14:30-15:00  Perguntas ou Demonstração ao Vivo (opcional)

═══════════════════════════════════════════════════════════════════════════
PARTE 7: ENTREGÁVEIS FINAIS
═══════════════════════════════════════════════════════════════════════════

16. CHECKLIST DE ENTREGÁVEIS:

    CÓDIGO E DADOS:
    ✓ Código completo (notebooks ou scripts Python)
    ✓ Dataset original + processado (CSV)
    ✓ Datasets perturbados (typos, sem emojis, etc.)
    ✓ requirements.txt
    ✓ README com instruções completas
    
    RESULTADOS:
    ✓ CSVs com métricas de TODAS as simulações
    ✓ Relatório estatístico (Wilcoxon, Kruskal-Wallis)
    ✓ Tabelas de análises avançadas (comprimento, typos, etc.)
    
    VISUALIZAÇÕES:
    ✓ Boxplots (Accuracy, Precision, Recall, F1)
    ✓ Gráficos de linha (evolução por simulação)
    ✓ Matrizes de confusão (agregadas)
    ✓ Gráficos de análises avançadas (comprimento, typos, emojis, etc.)
    ✓ Tabela de p-valores (significância estatística)
    
    ANÁLISES:
    ✓ Documento com exemplos de acertos/erros (20-30 exemplos anotados)
    ✓ Relatório de análise de comprimento
    ✓ Relatório de robustez a typos
    ✓ Relatório de análise de emojis
    ✓ Relatório de análise de sarcasmo
    ✓ Relatório de análise de formalidade
    
    APRESENTAÇÃO:
    ✓ Slides (PDF + PPTX)
    ✓ Vídeo (máximo 15 minutos)
    ✓ Script/roteiro do vídeo
    ✓ Material suplementar (se necessário)

═══════════════════════════════════════════════════════════════════════════
PARTE 8: CRONOGRAMA AJUSTADO (2-3 SEMANAS)
═══════════════════════════════════════════════════════════════════════════

17. PLANO DE EXECUÇÃO:

    SEMANA 1 - IMPLEMENTAÇÃO BÁSICA:
    Dia 1-2:  Coleta de dados + EDA (4h)
    Dia 3:    Preparar datasets perturbados (2h)
    Dia 4:    SVM + BoW (2h)
    Dia 5:    SVM + Embeddings (3h)
    Dia 6-7:  BERT fine-tuning (4h)
    
    SEMANA 2 - EXPERIMENTOS E VALIDAÇÃO:
    Dia 8:    LLM in-context (2h)
    Dia 9-10: Executar 10-30 simulações de cada modelo (6h)
    Dia 11:   Testes estatísticos (Wilcoxon, Kruskal-Wallis) (2h)
    Dia 12:   Análise de erros qualitativos (3h)
    Dia 13:   Análise de comprimento + typos (3h)
    Dia 14:   Análise de emojis + sarcasmo + formalidade (4h)
    
    SEMANA 3 - APRESENTAÇÃO:
    Dia 15-16: Gerar todas as visualizações (4h)
    Dia 17-18: Preparar slides (4h)
    Dia 19:    Escrever roteiro do vídeo (2h)
    Dia 20:    Gravar e editar vídeo (3h)
    Dia 21:    Revisão final + ajustes (2h)
    
    TOTAL: ~50-60 horas de trabalho

═══════════════════════════════════════════════════════════════════════════
PARTE 9: DIFERENCIAL DESTE PROJETO
═══════════════════════════════════════════════════════════════════════════

18. O QUE TORNA ESTE PROJETO EXCEPCIONAL:

    RIGOR CIENTÍFICO:
    ✓ Validação estatística com Wilcoxon (α=0.05)
    ✓ 10-30 simulações (robustez)
    ✓ Múltiplas métricas (não só accuracy)
    ✓ Visualizações profissionais (boxplots, linhas, heatmaps)
    
    ANÁLISES AVANÇADAS DE NLP:
    ✓ Comprimento de texto vs performance
    ✓ Robustez a typos (perturbação controlada)
    ✓ Análise de emojis (informatividade)
    ✓ Detecção de sarcasmo/ironia (desafio conhecido)
    ✓ Sensibilidade a formalidade/dialeto
    
    ANÁLISE QUALITATIVA:
    ✓ 20-30 exemplos anotados de acertos/erros
    ✓ Identificação de padrões de erro
    ✓ Insights linguísticos específicos
    
    ABORDAGEM MODERNA:
    ✓ 3 modelos clássicos + 1 estado-da-arte (LLM)
    ✓ In-context learning (tendência atual)
    ✓ Comparação justa (mesmos dados, mesmas métricas)
    
    APRESENTAÇÃO IMPACTANTE:
    ✓ Slides profissionais com dados reais
    ✓ Vídeo bem estruturado (15 min)
    ✓ Análises práticas e acionáveis
    ✓ Material suplementar completo

═══════════════════════════════════════════════════════════════════════════

Por favor, desenvolva este projeto COMPLETO com:
1. Código modular e bem documentado
2. Validação estatística RIGOROSA (Wilcoxon, múltiplas simulações)
3. Análises avançadas ESPECÍFICAS de NLP (comprimento, typos, emojis, 
   sarcasmo, formalidade)
4. Visualizações profissionais para apresentação
5. Documentação detalhada de todos os experimentos

Este projeto combina SIMPLICIDADE DE EXECUÇÃO (modelos estabelecidos) com
RIGOR CIENTÍFICO (validação estatística + análises profundas), adequado
para disciplina de mestrado ou publicação em workshop de NLP.