# 📊 Guia Visual do Sistema

> Entenda o sistema através de diagramas visuais

---

## 🎯 Jornada do Usuário

### Cenário 1: Treinar e Usar pela Primeira Vez

```mermaid
journey
    title Primeira Vez no Sistema
    section Setup
      Clonar repositório: 5: Usuário
      Instalar dependências: 4: Usuário
      Configurar ambiente: 3: Usuário
    section Treinamento
      Executar train_model.py: 5: Usuário
      Aguardar treinamento: 3: Usuário, Sistema
      Ver métricas: 5: Usuário
    section Uso
      Iniciar API: 5: Usuário
      Fazer previsão: 5: Usuário
      Ver resultado: 5: Usuário
```

### Cenário 2: Desenvolvedor Contribuindo

```mermaid
journey
    title Workflow de Desenvolvimento
    section Desenvolvimento
      Criar branch: 5: Dev
      Implementar feature: 4: Dev
      Rodar testes locais: 5: Dev
    section Review
      Abrir PR: 5: Dev
      CI/CD roda testes: 4: Sistema
      Code review: 4: Team
    section Deploy
      Merge para main: 5: Dev
      Deploy automático: 5: Sistema
      Verificar produção: 5: Dev
```

---

## 🏗️ Como os Dados Fluem no Sistema

### Do Yahoo Finance até a Previsão

```mermaid
graph LR
    A[📈 Yahoo<br/>Finance] -->|Raw Data| B[🔄 Data<br/>Loader]
    B -->|DataFrame| C[🎨 Feature<br/>Engineering]
    C -->|16 Features| D[📏 Min-Max<br/>Scaler]
    D -->|Normalized| E[🔲 Sequence<br/>Creator]
    E -->|60 days| F[🧠 LSTM<br/>Model]
    F -->|Scaled Pred| G[🔄 Inverse<br/>Transform]
    G -->|$ Price| H[📤 API<br/>Response]
    
    style A fill:#e1f5ff
    style F fill:#fff4e1
    style H fill:#e8f5e9
```

### Transformação dos Dados (Detalhado)

```mermaid
graph TB
    subgraph "1. Raw Data"
        RAW["Date | Open | High | Low | Close | Volume<br/>2024-01-01 | 180.50 | 182.30 | 179.80 | 181.90 | 52.3M"]
    end
    
    subgraph "2. Feature Engineering"
        FE["Date | Close | MA_7 | MA_30 | Volatility | Momentum | ...<br/>2024-01-01 | 181.90 | 180.45 | 175.32 | 2.45 | 3.20 | ..."]
    end
    
    subgraph "3. Normalization"
        NORM["Date | Close_norm | MA_7_norm | MA_30_norm | ...<br/>2024-01-01 | 0.856 | 0.823 | 0.745 | ..."]
    end
    
    subgraph "4. Sequences"
        SEQ["[Day 1, Day 2, ..., Day 60] → [Day 61 Prediction]<br/>Shape: (60, 16) → (1,)"]
    end
    
    subgraph "5. Model Output"
        OUT["Normalized Prediction: 0.872<br/>↓ Inverse Transform ↓<br/>Actual Price: $185.50"]
    end
    
    RAW --> FE
    FE --> NORM
    NORM --> SEQ
    SEQ --> OUT
    
    style RAW fill:#e1f5ff
    style FE fill:#fff3e0
    style NORM fill:#f3e5f5
    style SEQ fill:#e8f5e9
    style OUT fill:#fff4e1
```

---

## 🧠 Anatomia do Modelo LSTM

### Estrutura Visual

```mermaid
graph TB
    subgraph "Input"
        IN["60 dias × 16 features<br/>= Matrix (60, 16)"]
    end
    
    subgraph "LSTM Layer 1"
        L1_CELL1["LSTM Cell<br/>Hidden: 50"]
        L1_CELL2["LSTM Cell<br/>Hidden: 50"]
        L1_CELLN["..."]
        L1_CELL60["LSTM Cell<br/>Hidden: 50"]
    end
    
    subgraph "Dropout 20%"
        DROP1["Regularização<br/>Previne Overfitting"]
    end
    
    subgraph "LSTM Layer 2"
        L2_CELL1["LSTM Cell<br/>Hidden: 50"]
        L2_CELL2["LSTM Cell<br/>Hidden: 50"]
        L2_CELLN["..."]
        L2_CELL60["LSTM Cell<br/>Hidden: 50"]
    end
    
    subgraph "Dropout 20%"
        DROP2["Regularização"]
    end
    
    subgraph "Fully Connected"
        FC["Linear Layer<br/>50 → 1"]
    end
    
    subgraph "Output"
        OUT["Preço Previsto<br/>$ 185.50"]
    end
    
    IN --> L1_CELL1
    L1_CELL1 --> L1_CELL2
    L1_CELL2 --> L1_CELLN
    L1_CELLN --> L1_CELL60
    L1_CELL60 --> DROP1
    
    DROP1 --> L2_CELL1
    L2_CELL1 --> L2_CELL2
    L2_CELL2 --> L2_CELLN
    L2_CELLN --> L2_CELL60
    L2_CELL60 --> DROP2
    
    DROP2 --> FC
    FC --> OUT
    
    style IN fill:#e1f5ff
    style L1_CELL60 fill:#fff4e1
    style L2_CELL60 fill:#fff4e1
    style OUT fill:#e8f5e9
```

### Como o LSTM "Lembra"

```mermaid
graph LR
    subgraph "Dia 1"
        D1[Open: 180<br/>Close: 182<br/>Volume: 50M]
        H1[Hidden State<br/>50 valores]
        C1[Cell State<br/>50 valores]
    end
    
    subgraph "Dia 2"
        D2[Open: 182<br/>Close: 184<br/>Volume: 52M]
        H2[Hidden State<br/>Atualizado]
        C2[Cell State<br/>Atualizado]
    end
    
    subgraph "..."
        DN[...]
    end
    
    subgraph "Dia 60"
        D60[Open: 180<br/>Close: 183<br/>Volume: 55M]
        H60[Hidden State<br/>Final]
        C60[Cell State<br/>Memória Acumulada]
    end
    
    D1 --> H1
    D1 --> C1
    H1 --> D2
    C1 --> D2
    
    D2 --> H2
    D2 --> C2
    H2 --> DN
    C2 --> DN
    
    DN --> D60
    D60 --> H60
    D60 --> C60
    
    H60 -->|Usado para| PRED[Previsão<br/>Dia 61]
    C60 -->|Memória de| PRED
    
    style D1 fill:#e1f5ff
    style C60 fill:#fff4e1
    style PRED fill:#e8f5e9
```

---

## 🔄 Estado do Sistema

### Ciclo de Vida de um Modelo

```mermaid
stateDiagram-v2
    [*] --> NotTrained: Modelo não existe
    
    NotTrained --> Training: user executa train_model.py
    Training --> Validating: após 50 epochs
    Validating --> Trained: métricas OK
    Validating --> Failed: métricas ruins
    
    Failed --> NotTrained: recomeçar
    
    Trained --> InProduction: modelo salvo e carregado na API
    InProduction --> Serving: recebendo requests
    
    Serving --> Monitoring: coletando métricas
    Monitoring --> Serving: performance OK
    Monitoring --> Retraining: performance degrada
    
    Retraining --> Training: retreinar com dados novos
    
    InProduction --> Deprecated: novo modelo treinado
    Deprecated --> [*]
```

### Estados da API

```mermaid
stateDiagram-v2
    [*] --> Starting: uvicorn starts
    
    Starting --> LoadingModels: carrega modelos disponíveis
    LoadingModels --> Ready: modelos carregados
    LoadingModels --> PartialReady: alguns modelos falharam
    
    Ready --> Serving: recebe requests
    PartialReady --> Serving: serve apenas modelos OK
    
    Serving --> Processing: processa request
    Processing --> Serving: retorna response
    
    Serving --> HealthCheck: /health endpoint
    HealthCheck --> Serving: status OK
    
    Serving --> Shutdown: SIGTERM
    Shutdown --> [*]
```

---

## 📊 Métricas Visuais

### O que significa cada métrica?

```mermaid
graph TB
    subgraph "RMSE - Root Mean Square Error"
        RMSE1["Penaliza erros grandes<br/>mais que pequenos"]
        RMSE2["Em dólares ($)<br/>Ex: RMSE = 3.45"]
        RMSE3["Quanto menor melhor<br/>< 5% do preço é bom"]
    end
    
    subgraph "MAE - Mean Absolute Error"
        MAE1["Erro médio absoluto<br/>mais intuitivo"]
        MAE2["Em dólares ($)<br/>Ex: MAE = 2.67"]
        MAE3["Não penaliza outliers<br/>tanto quanto RMSE"]
    end
    
    subgraph "MAPE - Mean Absolute % Error"
        MAPE1["Erro em porcentagem<br/>fácil de interpretar"]
        MAPE2["Ex: MAPE = 1.89%<br/>= erro de ~2%"]
        MAPE3["< 10% é excelente<br/>10-20% é bom"]
    end
    
    subgraph "R² - Coefficient of Determination"
        R21["% da variância explicada<br/>pelo modelo"]
        R22["Varia de 0 a 1<br/>Ex: R² = 0.9567"]
        R23["0.9-1.0 é excelente<br/>explica 95.67%"]
    end
    
    style RMSE1 fill:#ffebee
    style MAE1 fill:#e3f2fd
    style MAPE1 fill:#f3e5f5
    style R21 fill:#e8f5e9
```

### Exemplo Real de Avaliação

```mermaid
graph LR
    subgraph "Modelo Treinado"
        M["LSTM AAPL<br/>50 epochs<br/>1356 amostras"]
    end
    
    subgraph "Test Set (340 amostras)"
        T["Últimos 340 dias<br/>não vistos no treino"]
    end
    
    subgraph "Previsões"
        P["340 previsões<br/>vs 340 valores reais"]
    end
    
    subgraph "Métricas"
        ME["RMSE: $3.45<br/>MAE: $2.67<br/>MAPE: 1.89%<br/>R²: 0.9567<br/>Dir Acc: 76.47%"]
    end
    
    subgraph "Interpretação"
        I["✅ Excelente<br/>Erro médio < 2%<br/>Acerta direção em 3/4"]
    end
    
    M --> T
    T --> P
    P --> ME
    ME --> I
    
    style M fill:#fff4e1
    style ME fill:#e1f5ff
    style I fill:#e8f5e9
```

---

## 🌐 Arquitetura de Deploy

### Development Environment

```mermaid
graph TB
    subgraph "Your Computer"
        CODE[Código-fonte]
        VENV[Virtual Env<br/>Python 3.10]
        JUPYTER[Jupyter Notebook<br/>Explorações]
    end
    
    subgraph "Local Services"
        API[FastAPI<br/>localhost:8000]
        GRADIO[Gradio UI<br/>localhost:7860]
    end
    
    subgraph "External"
        YAHOO[Yahoo Finance<br/>Dados]
    end
    
    CODE --> VENV
    VENV --> API
    VENV --> GRADIO
    VENV --> JUPYTER
    API --> YAHOO
    GRADIO --> API
    
    style CODE fill:#e1f5ff
    style API fill:#e8f5e9
    style YAHOO fill:#fff4e1
```

### Production Environment

```mermaid
graph TB
    subgraph "GitHub"
        REPO[Repositório]
        ACTIONS[GitHub Actions<br/>CI/CD]
    end
    
    subgraph "Docker"
        DOCKER[Docker Image<br/>Multi-stage build]
    end
    
    subgraph "Railway Cloud"
        RAIL1[Container 1<br/>US-West]
        RAIL2[Container 2<br/>US-East]
        LB[Load Balancer]
    end
    
    subgraph "HuggingFace"
        HF[Gradio UI<br/>Demo]
    end
    
    subgraph "Users"
        USER1[User 1]
        USER2[User 2]
        USER3[User 3]
    end
    
    REPO --> ACTIONS
    ACTIONS --> DOCKER
    DOCKER --> RAIL1
    DOCKER --> RAIL2
    
    USER1 --> LB
    USER2 --> LB
    USER3 --> HF
    
    LB --> RAIL1
    LB --> RAIL2
    
    HF --> LB
    
    style REPO fill:#e1f5ff
    style DOCKER fill:#fff4e1
    style RAIL1 fill:#e8f5e9
    style HF fill:#f3e5f5
```

---

## 🔍 Troubleshooting Visual

### Diagnóstico de Problemas

```mermaid
graph TB
    START{Problema?}
    
    START -->|API não inicia| CHECK1{Porta em uso?}
    CHECK1 -->|Sim| SOL1[Mudar porta<br/>--port 8001]
    CHECK1 -->|Não| CHECK2{Deps instaladas?}
    CHECK2 -->|Não| SOL2[pip install -r<br/>requirements.txt]
    CHECK2 -->|Sim| SOL3[Checar logs<br/>ver erro exato]
    
    START -->|Model not found| CHECK3{Modelo treinado?}
    CHECK3 -->|Não| SOL4[Treinar modelo<br/>train_model.py AAPL]
    CHECK3 -->|Sim| CHECK4{Path correto?}
    CHECK4 -->|Não| SOL5[Verificar<br/>models/ dir]
    
    START -->|Previsão ruim| CHECK5{MAPE > 20%?}
    CHECK5 -->|Sim| SOL6[Retreinar com<br/>mais dados]
    CHECK5 -->|Não| CHECK6{Dir Acc < 60%?}
    CHECK6 -->|Sim| SOL7[Ajustar<br/>hiperparâmetros]
    CHECK6 -->|Não| SOL8[Performance OK<br/>é esperado]
    
    START -->|API lenta| CHECK7{> 1s latência?}
    CHECK7 -->|Sim| SOL9[Aumentar workers<br/>ou usar cache]
    CHECK7 -->|Não| SOL10[Performance OK]
    
    style START fill:#fff4e1
    style SOL1 fill:#e8f5e9
    style SOL2 fill:#e8f5e9
    style SOL4 fill:#e8f5e9
    style SOL6 fill:#e8f5e9
    style SOL9 fill:#e8f5e9
```

---

## 📈 Evolução do Sistema

### Roadmap Visual

```mermaid
timeline
    title Evolução do Projeto
    
    section Fase 1 - MVP
        Semana 1 : Coleta de dados : yfinance
        Semana 2 : Modelo LSTM básico : PyTorch
        Semana 3 : API simples : FastAPI
    
    section Fase 2 - Melhorias
        Semana 4 : Feature engineering : 16 features
        Semana 5 : Testes automatizados : pytest
        Semana 6 : Logging e métricas : Loguru + Prometheus
    
    section Fase 3 - Deploy
        Semana 7 : Dockerização : Dockerfile + docker-compose
        Semana 8 : CI/CD : GitHub Actions
        Semana 9 : Deploy produção : Railway + HuggingFace
    
    section Fase 4 - Futuro
        Futuro : Banco de dados : PostgreSQL
               : Mais modelos : Transformer, Prophet
               : Cache : Redis
               : Monitoring avançado : Grafana
```

---

## 🎨 Paleta de Cores do Sistema

### Código de Cores para Diagramas

```mermaid
graph LR
    A[Entrada/Dados<br/>🔵 #e1f5ff] 
    B[Processamento<br/>🟡 #fff4e1]
    C[Saída/API<br/>🟢 #e8f5e9]
    D[Configuração<br/>🟣 #f3e5f5]
    E[Erro/Atenção<br/>🔴 #ffebee]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#e8f5e9
    style D fill:#f3e5f5
    style E fill:#ffebee
```

---

## 📚 Glossário Visual

### Termos-Chave Ilustrados

```mermaid
mindmap
  root((Sistema de<br/>Previsão))
    Dados
      Yahoo Finance
        API gratuita
        Dados históricos OHLCV
      Features
        16 indicadores técnicos
        Normalizadas 0-1
    Modelo
      LSTM
        2 camadas
        50 hidden units
        Dropout 0.2
      Training
        50 epochs
        Batch 32
        Adam optimizer
    API
      FastAPI
        REST endpoints
        Swagger docs
      Monitoramento
        Prometheus metrics
        Loguru logs
    Deploy
      Docker
        Multi-stage build
        Health checks
      Cloud
        Railway backend
        HuggingFace UI
```

---

**Este guia visual complementa o README principal**

*Use este documento para apresentações e explicações visuais*

*Última atualização: Dezembro 2024*

