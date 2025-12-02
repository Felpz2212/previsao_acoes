# 📊 Diagrama da Solução Completa

> Sistema de Previsão de Preços de Ações com LSTM - Tech Challenge Fase 4

---

## 🎯 Arquitetura Geral do Sistema

```mermaid
flowchart TB
    subgraph EXTERNAL["🌐 Fontes Externas"]
        YF["📈 Yahoo Finance<br/>API de Dados"]
    end

    subgraph DATA_LAYER["📊 Camada de Dados"]
        DL["🔄 Data Loader<br/>yfinance"]
        VAL["✅ Validação<br/>Qualidade dos Dados"]
        FE["🎨 Feature Engineering<br/>16 Features Técnicas"]
        PREP["📏 Preprocessor<br/>MinMaxScaler"]
        SEQ["🔲 Sequence Creator<br/>60 dias de janela"]
    end

    subgraph ML_LAYER["🧠 Camada de Machine Learning"]
        LSTM["🔮 LSTM Model<br/>PyTorch<br/>2 camadas × 50 neurons"]
        TRAIN["🏋️ Trainer<br/>50 epochs<br/>Adam optimizer"]
        EVAL["📊 Evaluator<br/>RMSE, MAE, MAPE, R²"]
    end

    subgraph STORAGE["💾 Armazenamento"]
        MODEL_FILE["📦 Modelo<br/>lstm_model_SYMBOL.pth"]
        SCALER_FILE["📦 Scaler<br/>scaler_SYMBOL.pkl"]
        META["📋 Metadata<br/>metadata_SYMBOL.json"]
    end

    subgraph API_LAYER["🌐 Camada de API"]
        FASTAPI["⚡ FastAPI<br/>REST API"]
        
        subgraph ROUTES["📍 Routes"]
            R_PRED["POST /predict<br/>Previsões"]
            R_DATA["GET /stocks<br/>Dados Históricos"]
            R_MODEL["POST /models<br/>Gerenciamento"]
            R_HEALTH["GET /health<br/>Monitoramento"]
        end
        
        VALID["✅ Pydantic<br/>Validação"]
        MW["🔧 Middlewares<br/>CORS, Metrics, Logging"]
    end

    subgraph MONITORING["📈 Monitoramento"]
        PROM["📊 Prometheus<br/>Métricas"]
        LOG["📝 Loguru<br/>Logs Estruturados"]
    end

    subgraph INFRA["🏗️ Infraestrutura"]
        DOCKER["🐳 Docker<br/>Container"]
        COMPOSE["🐙 Docker Compose<br/>Orquestração"]
        CICD["🔄 GitHub Actions<br/>CI/CD"]
    end

    subgraph DEPLOY["🚀 Deploy"]
        RAILWAY["🚂 Railway<br/>Backend API"]
        HF["🤗 HuggingFace<br/>UI Demo Gradio"]
    end

    subgraph CLIENTS["👥 Clientes"]
        WEB["🌐 Web Browser"]
        CLI["💻 CLI / cURL"]
        GRADIO["🎨 Gradio UI"]
    end

    %% Fluxo de Dados
    YF --> DL
    DL --> VAL
    VAL --> FE
    FE --> PREP
    PREP --> SEQ
    SEQ --> LSTM
    LSTM --> TRAIN
    TRAIN --> EVAL
    
    %% Salvamento
    EVAL --> MODEL_FILE
    EVAL --> SCALER_FILE
    EVAL --> META
    
    %% API
    MODEL_FILE --> FASTAPI
    SCALER_FILE --> FASTAPI
    FASTAPI --> MW
    MW --> VALID
    VALID --> ROUTES
    R_PRED --> PROM
    R_HEALTH --> PROM
    FASTAPI --> LOG
    
    %% Infraestrutura
    FASTAPI --> DOCKER
    DOCKER --> COMPOSE
    COMPOSE --> CICD
    CICD --> RAILWAY
    CICD --> HF
    
    %% Clientes
    WEB --> RAILWAY
    CLI --> RAILWAY
    GRADIO --> HF

    %% Estilos
    style YF fill:#e3f2fd,stroke:#1976d2
    style LSTM fill:#fff3e0,stroke:#f57c00
    style FASTAPI fill:#e8f5e9,stroke:#388e3c
    style RAILWAY fill:#f3e5f5,stroke:#7b1fa2
    style HF fill:#fce4ec,stroke:#c2185b
```

---

## 🔄 Fluxo de Treinamento Detalhado

```mermaid
flowchart LR
    subgraph INPUT["📥 Entrada"]
        CMD["python train_model.py AAPL"]
    end

    subgraph COLLECT["1️⃣ Coleta"]
        YAHOO["Yahoo Finance"]
        RAW["Dados OHLCV<br/>1756 registros"]
    end

    subgraph PROCESS["2️⃣ Processamento"]
        FEAT["Feature Engineering"]
        FEATURES["16 Features:<br/>• Preços (5)<br/>• Variações (4)<br/>• MAs (3)<br/>• Volatilidade (2)<br/>• Momentum (2)"]
        NORM["Normalização<br/>MinMaxScaler<br/>Range: 0-1"]
    end

    subgraph SEQUENCE["3️⃣ Sequências"]
        SEQ["Criar Janelas<br/>60 dias cada"]
        SPLIT["Split<br/>80% Train<br/>10% Val<br/>10% Test"]
    end

    subgraph TRAIN["4️⃣ Treinamento"]
        INIT["Inicializar LSTM<br/>input: 16<br/>hidden: 50<br/>layers: 2"]
        LOOP["Training Loop<br/>50 epochs"]
        OPT["Adam Optimizer<br/>LR: 0.001"]
    end

    subgraph EVAL["5️⃣ Avaliação"]
        METRICS["Calcular Métricas:<br/>• RMSE: 3.45<br/>• MAE: 2.67<br/>• MAPE: 1.89%<br/>• R²: 0.9567<br/>• Dir Acc: 76.47%"]
    end

    subgraph SAVE["6️⃣ Salvamento"]
        MODEL["lstm_model_AAPL.pth"]
        SCALER["scaler_AAPL.pkl"]
        META["metadata_AAPL.json"]
    end

    subgraph OUTPUT["📤 Saída"]
        DONE["✅ Modelo Pronto!"]
    end

    CMD --> YAHOO
    YAHOO --> RAW
    RAW --> FEAT
    FEAT --> FEATURES
    FEATURES --> NORM
    NORM --> SEQ
    SEQ --> SPLIT
    SPLIT --> INIT
    INIT --> LOOP
    LOOP --> OPT
    OPT --> METRICS
    METRICS --> MODEL
    METRICS --> SCALER
    METRICS --> META
    MODEL --> DONE
    SCALER --> DONE
    META --> DONE

    style CMD fill:#e8f5e9
    style LSTM fill:#fff3e0
    style METRICS fill:#e3f2fd
    style DONE fill:#c8e6c9
```

---

## 🔮 Fluxo de Predição

```mermaid
flowchart LR
    subgraph REQUEST["📥 Request"]
        REQ["POST /api/v1/predict<br/>{symbol: AAPL, days_ahead: 1}"]
    end

    subgraph VALIDATE["1️⃣ Validação"]
        PYDANTIC["Pydantic<br/>Validar Schema"]
        CHECK["Verificar<br/>Modelo Existe"]
    end

    subgraph LOAD["2️⃣ Carregar"]
        LOAD_MODEL["Carregar<br/>lstm_model_AAPL.pth"]
        LOAD_SCALER["Carregar<br/>scaler_AAPL.pkl"]
    end

    subgraph DATA["3️⃣ Dados"]
        YAHOO["Yahoo Finance<br/>Últimos 60 dias"]
        PROCESS["Criar Features<br/>Normalizar"]
    end

    subgraph PREDICT["4️⃣ Predição"]
        LSTM["LSTM Forward Pass<br/>Input: [1, 60, 16]"]
        INVERSE["Inverse Transform<br/>Desnormalizar"]
    end

    subgraph RESPONSE["📤 Response"]
        RESP["{<br/>  predicted_price: 185.50<br/>  current_price: 183.20<br/>  change_pct: 1.25%<br/>  prediction_date: 2024-12-03<br/>}"]
    end

    REQ --> PYDANTIC
    PYDANTIC --> CHECK
    CHECK --> LOAD_MODEL
    CHECK --> LOAD_SCALER
    LOAD_MODEL --> YAHOO
    LOAD_SCALER --> YAHOO
    YAHOO --> PROCESS
    PROCESS --> LSTM
    LSTM --> INVERSE
    INVERSE --> RESP

    style REQ fill:#e3f2fd
    style LSTM fill:#fff3e0
    style RESP fill:#e8f5e9
```

---

## 🧠 Arquitetura do Modelo LSTM

```mermaid
flowchart TB
    subgraph INPUT["📥 Input Layer"]
        IN["Sequência de Entrada<br/>Shape: [batch, 60, 16]<br/>60 dias × 16 features"]
    end

    subgraph LSTM1["🔄 LSTM Layer 1"]
        L1["LSTM<br/>hidden_size: 50<br/>bidirectional: false"]
        D1["Dropout: 20%"]
    end

    subgraph LSTM2["🔄 LSTM Layer 2"]
        L2["LSTM<br/>hidden_size: 50<br/>bidirectional: false"]
        D2["Dropout: 20%"]
    end

    subgraph OUTPUT["📤 Output Layer"]
        LAST["Último Timestep<br/>Shape: [batch, 50]"]
        FC["Fully Connected<br/>Linear(50 → 1)"]
        OUT["Previsão<br/>Shape: [batch, 1]"]
    end

    IN --> L1
    L1 --> D1
    D1 --> L2
    L2 --> D2
    D2 --> LAST
    LAST --> FC
    FC --> OUT

    style IN fill:#e3f2fd
    style L1 fill:#fff3e0
    style L2 fill:#fff3e0
    style OUT fill:#e8f5e9
```

---

## 🎨 Features do Modelo

```mermaid
mindmap
    root((16 Features))
        Preços Base
            Open
            High
            Low
            Close
            Volume
        Variações %
            price_change
            high_low_pct
            close_open_pct
            volume_change
        Médias Móveis
            MA 7 dias
            MA 30 dias
            MA 90 dias
        Volatilidade
            Vol 7 dias
            Vol 30 dias
        Momentum
            Momentum 4d
            Volume MA 7d
```

---

## 🌐 Arquitetura da API

```mermaid
flowchart TB
    subgraph CLIENTS["👥 Clientes"]
        BROWSER["🌐 Browser"]
        CURL["💻 cURL"]
        PYTHON["🐍 Python"]
    end

    subgraph GATEWAY["🚪 API Gateway"]
        FASTAPI["⚡ FastAPI<br/>v1.0.0"]
        CORS["🔒 CORS"]
        METRICS_MW["📊 Metrics MW"]
        TIMING["⏱️ Timing MW"]
    end

    subgraph ROUTES["📍 Endpoints"]
        subgraph PREDICTIONS["Predictions"]
            POST_PRED["POST /predict"]
            POST_BATCH["POST /predict/batch"]
        end
        
        subgraph DATA_ROUTES["Data"]
            GET_HIST["GET /stocks/{symbol}/historical"]
            GET_LATEST["GET /stocks/{symbol}/latest"]
            GET_AVAIL["GET /stocks/available"]
        end
        
        subgraph MODEL_ROUTES["Models"]
            POST_TRAIN["POST /models/train"]
            GET_STATUS["GET /models/status"]
            GET_PERF["GET /models/{symbol}/performance"]
        end
        
        subgraph MONITORING["Monitoring"]
            GET_HEALTH["GET /health"]
            GET_METRICS["GET /metrics"]
            GET_PROM["GET /metrics/prometheus"]
        end
    end

    subgraph SERVICES["⚙️ Services"]
        PRED_SVC["Prediction Service"]
        DATA_SVC["Data Service"]
        MODEL_SVC["Model Service"]
        MON_SVC["Monitoring Service"]
    end

    subgraph EXTERNAL["🌐 External"]
        YAHOO["Yahoo Finance"]
        FILES["File System"]
    end

    BROWSER --> FASTAPI
    CURL --> FASTAPI
    PYTHON --> FASTAPI
    
    FASTAPI --> CORS
    CORS --> METRICS_MW
    METRICS_MW --> TIMING
    
    TIMING --> POST_PRED
    TIMING --> POST_BATCH
    TIMING --> GET_HIST
    TIMING --> GET_LATEST
    TIMING --> GET_AVAIL
    TIMING --> POST_TRAIN
    TIMING --> GET_STATUS
    TIMING --> GET_PERF
    TIMING --> GET_HEALTH
    TIMING --> GET_METRICS
    TIMING --> GET_PROM
    
    POST_PRED --> PRED_SVC
    POST_BATCH --> PRED_SVC
    GET_HIST --> DATA_SVC
    GET_LATEST --> DATA_SVC
    GET_AVAIL --> DATA_SVC
    POST_TRAIN --> MODEL_SVC
    GET_STATUS --> MODEL_SVC
    GET_PERF --> MODEL_SVC
    GET_HEALTH --> MON_SVC
    GET_METRICS --> MON_SVC
    GET_PROM --> MON_SVC
    
    DATA_SVC --> YAHOO
    PRED_SVC --> FILES
    MODEL_SVC --> FILES

    style FASTAPI fill:#e8f5e9
    style PRED_SVC fill:#fff3e0
    style YAHOO fill:#e3f2fd
```

---

## 🚀 Pipeline de Deploy

```mermaid
flowchart LR
    subgraph DEV["💻 Development"]
        CODE["📝 Código"]
        TEST_LOCAL["🧪 Testes Locais"]
        COMMIT["📦 Git Commit"]
    end

    subgraph CICD["🔄 CI/CD"]
        PUSH["📤 Git Push"]
        GH_ACTIONS["⚙️ GitHub Actions"]
        
        subgraph PIPELINE["Pipeline"]
            LINT["🔍 Lint<br/>ruff"]
            FORMAT["📐 Format<br/>black"]
            TYPE["📝 Type Check<br/>mypy"]
            PYTEST["🧪 Tests<br/>pytest"]
            BUILD["🐳 Build Docker"]
        end
    end

    subgraph DEPLOY_TARGET["🚀 Deploy"]
        RAILWAY["🚂 Railway<br/>Backend API"]
        HF["🤗 HuggingFace<br/>Gradio UI"]
    end

    subgraph PROD["🌐 Production"]
        API_LIVE["⚡ API Live<br/>api.railway.app"]
        UI_LIVE["🎨 UI Live<br/>hf.co/spaces"]
        HEALTH["❤️ Health Checks"]
        LOGS["📝 Logs"]
        METRICS_PROD["📊 Metrics"]
    end

    CODE --> TEST_LOCAL
    TEST_LOCAL --> COMMIT
    COMMIT --> PUSH
    PUSH --> GH_ACTIONS
    GH_ACTIONS --> LINT
    LINT --> FORMAT
    FORMAT --> TYPE
    TYPE --> PYTEST
    PYTEST --> BUILD
    BUILD --> RAILWAY
    BUILD --> HF
    RAILWAY --> API_LIVE
    HF --> UI_LIVE
    API_LIVE --> HEALTH
    API_LIVE --> LOGS
    API_LIVE --> METRICS_PROD

    style CODE fill:#e3f2fd
    style GH_ACTIONS fill:#fff3e0
    style RAILWAY fill:#e8f5e9
    style HF fill:#fce4ec
```

---

## 📊 Métricas de Avaliação

```mermaid
flowchart TB
    subgraph MODEL["🧠 Modelo Treinado"]
        PRED["Previsões<br/>340 amostras"]
        REAL["Valores Reais<br/>340 amostras"]
    end

    subgraph METRICS["📊 Métricas Calculadas"]
        subgraph ERROR["Métricas de Erro"]
            RMSE["RMSE<br/>Root Mean Square Error<br/>$3.45"]
            MAE["MAE<br/>Mean Absolute Error<br/>$2.67"]
            MAPE["MAPE<br/>Mean Absolute % Error<br/>1.89%"]
        end
        
        subgraph FIT["Métricas de Ajuste"]
            R2["R²<br/>Coef. Determinação<br/>0.9567 (95.67%)"]
            DIR["Directional Accuracy<br/>Acurácia Direcional<br/>76.47%"]
        end
    end

    subgraph INTERPRET["📝 Interpretação"]
        GOOD["✅ Resultados:<br/>• Erro médio < 2%<br/>• Explica 95% variância<br/>• Acerta direção 3/4 vezes"]
    end

    PRED --> RMSE
    REAL --> RMSE
    PRED --> MAE
    REAL --> MAE
    PRED --> MAPE
    REAL --> MAPE
    PRED --> R2
    REAL --> R2
    PRED --> DIR
    REAL --> DIR
    
    RMSE --> GOOD
    MAE --> GOOD
    MAPE --> GOOD
    R2 --> GOOD
    DIR --> GOOD

    style PRED fill:#e3f2fd
    style MAPE fill:#c8e6c9
    style R2 fill:#c8e6c9
    style GOOD fill:#e8f5e9
```

---

## 🗂️ Estrutura do Projeto

```mermaid
flowchart TB
    subgraph ROOT["📁 previsao_acoes/"]
        README["📄 README.md"]
        DOCKER["🐳 Dockerfile"]
        COMPOSE["🐙 docker-compose.yml"]
        REQS["📦 requirements.txt"]
        
        subgraph SRC["📂 src/"]
            subgraph API["api/"]
                MAIN["main.py"]
                SCHEMAS["schemas.py"]
                ROUTES_DIR["routes/"]
            end
            
            subgraph DATA_DIR["data/"]
                LOADER["data_loader.py"]
                PREPROC["preprocessor.py"]
            end
            
            subgraph MODELS_DIR["models/"]
                LSTM_FILE["lstm_model.py"]
            end
            
            subgraph TRAIN_DIR["training/"]
                TRAINER["trainer.py"]
            end
            
            subgraph UTILS["utils/"]
                LOGGER["logger.py"]
                MONITORING["monitoring.py"]
            end
        end
        
        subgraph DOCS["📂 docs/"]
            DOC1["README_COMPLETO.md"]
            DOC2["GUIA_VISUAL.md"]
            DOC3["ARQUITETURA_TECNICA.md"]
            DOC4["+ 8 documentos"]
        end
        
        subgraph TESTS_DIR["📂 tests/"]
            T1["test_api.py"]
            T2["test_model.py"]
            T3["test_preprocessor.py"]
            T4["test_data_loader.py"]
        end
        
        subgraph SCRIPTS["📂 scripts/"]
            TRAIN_SCRIPT["train_model.py"]
        end
        
        subgraph CONFIG["📂 config/"]
            SETTINGS["settings.py"]
        end
    end

    style ROOT fill:#f5f5f5
    style API fill:#e8f5e9
    style MODELS_DIR fill:#fff3e0
    style DOCS fill:#e3f2fd
```

---

## 🔄 Ciclo de Vida do Modelo

```mermaid
stateDiagram-v2
    [*] --> NaoTreinado: Projeto iniciado
    
    NaoTreinado --> Treinando: train_model.py AAPL
    Treinando --> Validando: Após 50 epochs
    Validando --> Treinado: Métricas OK
    Validando --> Falhou: Métricas ruins
    
    Falhou --> NaoTreinado: Ajustar parâmetros
    
    Treinado --> EmProducao: Deploy API
    EmProducao --> Servindo: Recebendo requests
    
    Servindo --> Monitorando: Coletando métricas
    Monitorando --> Servindo: Performance OK
    Monitorando --> Retreinando: Performance degradou
    
    Retreinando --> Treinando: Com dados atualizados
    
    EmProducao --> Obsoleto: Novo modelo melhor
    Obsoleto --> [*]
```

---

## 🎯 Resumo Visual da Solução

```mermaid
graph TB
    subgraph PROBLEM["❓ Problema"]
        P1["Prever preços de ações<br/>é complexo"]
    end

    subgraph SOLUTION["💡 Solução"]
        S1["🧠 LSTM<br/>Deep Learning"]
        S2["📊 16 Features<br/>Indicadores Técnicos"]
        S3["⚡ FastAPI<br/>REST API"]
        S4["🐳 Docker<br/>Containerização"]
        S5["🚀 Railway<br/>Cloud Deploy"]
    end

    subgraph RESULT["✅ Resultado"]
        R1["MAPE: 1.89%<br/>Erro muito baixo"]
        R2["R²: 0.9567<br/>95% explicado"]
        R3["Dir Acc: 76%<br/>Acerta 3/4"]
        R4["API Production-Ready<br/>Escalável"]
        R5["Documentação Completa<br/>300+ páginas"]
    end

    P1 --> S1
    P1 --> S2
    S1 --> S3
    S2 --> S3
    S3 --> S4
    S4 --> S5
    
    S5 --> R1
    S5 --> R2
    S5 --> R3
    S5 --> R4
    S5 --> R5

    style P1 fill:#ffcdd2
    style S1 fill:#fff3e0
    style S3 fill:#e8f5e9
    style R1 fill:#c8e6c9
    style R5 fill:#bbdefb
```

---

## 📋 Tech Stack Completa

```mermaid
mindmap
    root((Tech Stack))
        ML & Data Science
            Python 3.10
            PyTorch 2.0
            NumPy
            Pandas
            scikit-learn
            yfinance
        API
            FastAPI
            Uvicorn
            Pydantic
        Monitoring
            Prometheus
            Loguru
        Testing
            pytest
            httpx
        DevOps
            Docker
            Docker Compose
            GitHub Actions
        Deploy
            Railway
            HuggingFace Spaces
        Documentation
            Markdown
            Mermaid Diagrams
```

---

## 📌 Links Importantes

| Recurso | Descrição |
|---------|-----------|
| 📊 **Yahoo Finance** | Fonte de dados |
| ⚡ **FastAPI** | Framework API |
| 🔥 **PyTorch** | Deep Learning |
| 🐳 **Docker** | Containerização |
| 🚂 **Railway** | Cloud Deploy |
| 🤗 **HuggingFace** | UI Demo |

---

## ⚠️ Disclaimer

> **Este é um projeto educacional** desenvolvido para o Tech Challenge Fase 4 da FIAP.
> 
> **NÃO USE** para decisões reais de investimento. O mercado de ações é altamente volátil e imprevisível.

---

*Tech Challenge Fase 4 - FIAP Pós-Tech Machine Learning Engineering*

*Dezembro 2024*

