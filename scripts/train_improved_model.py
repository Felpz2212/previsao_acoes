#!/usr/bin/env python3
"""
Script para treinar modelo LSTM melhorado.

Uso:
    python scripts/train_improved_model.py AAPL
    python scripts/train_improved_model.py AAPL --epochs 200 --patience 20
    python scripts/train_improved_model.py ALL  # Treina vários símbolos

Melhorias vs modelo original:
    - Early stopping para evitar overfitting
    - Learning rate scheduler adaptativo
    - Gradient clipping para estabilidade
    - LSTM bidirecional com attention
    - Walk-forward validation
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.training.improved_trainer import ImprovedModelTrainer


# Símbolos populares para treinamento
POPULAR_SYMBOLS = [
    # US Tech
    "AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA",
    # US Finance
    "JPM", "BAC", "V", "MA",
    # Brazil
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA"
]


def train_single_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    epochs: int,
    patience: int,
    batch_size: int
) -> dict:
    """Treina modelo para um único símbolo."""
    
    trainer = ImprovedModelTrainer(
        symbol=symbol,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        hidden_size=64,
        num_layers=3,
        dropout=0.3
    )
    
    try:
        metrics = trainer.run_training_pipeline(
            start_date=start_date,
            end_date=end_date,
            train_split=0.7,
            val_split=0.15
        )
        return {"success": True, "metrics": metrics}
    except Exception as e:
        logger.error(f"❌ Falha ao treinar {symbol}: {e}")
        return {"success": False, "error": str(e)}


def train_multiple_symbols(
    symbols: list,
    start_date: str,
    end_date: str,
    epochs: int,
    patience: int,
    batch_size: int
):
    """Treina modelos para múltiplos símbolos."""
    
    results = {}
    successful = 0
    failed = 0
    
    logger.info(f"🎯 Treinando {len(symbols)} símbolos...")
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(symbols)}] Treinando {symbol}...")
        logger.info(f"{'='*60}")
        
        result = train_single_symbol(
            symbol, start_date, end_date,
            epochs, patience, batch_size
        )
        results[symbol] = result
        
        if result["success"]:
            successful += 1
        else:
            failed += 1
    
    # Resumo final
    logger.info("\n" + "="*60)
    logger.info("📊 RESUMO DO TREINAMENTO EM LOTE")
    logger.info("="*60)
    logger.info(f"✅ Sucesso: {successful}/{len(symbols)}")
    logger.info(f"❌ Falhas: {failed}/{len(symbols)}")
    
    if successful > 0:
        logger.info("\n📈 Métricas por símbolo:")
        for symbol, result in results.items():
            if result["success"]:
                m = result["metrics"]
                logger.info(
                    f"  {symbol}: RMSE=${m.get('rmse',0):.2f}, "
                    f"MAPE={m.get('mape',0):.2f}%, "
                    f"R²={m.get('r2',0):.4f}, "
                    f"Dir={m.get('directional_accuracy',0):.1f}%"
                )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Treinar modelo LSTM melhorado para previsão de ações",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python scripts/train_improved_model.py AAPL
    python scripts/train_improved_model.py AAPL GOOGL MSFT
    python scripts/train_improved_model.py ALL --epochs 150
    python scripts/train_improved_model.py PETR4.SA --start 2020-01-01
        """
    )
    
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Símbolo(s) da ação. Use 'ALL' para símbolos populares"
    )
    parser.add_argument(
        "--start",
        default="2018-01-01",
        help="Data inicial (YYYY-MM-DD). Default: 2018-01-01"
    )
    parser.add_argument(
        "--end",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Data final (YYYY-MM-DD). Default: hoje"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número máximo de épocas. Default: 100"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Épocas sem melhoria antes de parar. Default: 15"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do batch. Default: 32"
    )
    
    args = parser.parse_args()
    
    # Determinar símbolos
    if "ALL" in [s.upper() for s in args.symbols]:
        symbols = POPULAR_SYMBOLS
    else:
        symbols = [s.upper() for s in args.symbols]
    
    # Configurar logging
    logger.info(f"🚀 Iniciando treinamento melhorado")
    logger.info(f"   Símbolos: {', '.join(symbols)}")
    logger.info(f"   Período: {args.start} até {args.end}")
    logger.info(f"   Config: epochs={args.epochs}, patience={args.patience}")
    
    # Treinar
    if len(symbols) == 1:
        result = train_single_symbol(
            symbols[0],
            args.start,
            args.end,
            args.epochs,
            args.patience,
            args.batch_size
        )
        
        if result["success"]:
            logger.info("✅ Treinamento concluído com sucesso!")
            sys.exit(0)
        else:
            logger.error(f"❌ Falha: {result['error']}")
            sys.exit(1)
    else:
        results = train_multiple_symbols(
            symbols,
            args.start,
            args.end,
            args.epochs,
            args.patience,
            args.batch_size
        )
        
        failed = sum(1 for r in results.values() if not r["success"])
        sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()

