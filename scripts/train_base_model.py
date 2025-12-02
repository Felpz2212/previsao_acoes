#!/usr/bin/env python3
"""
Script para treinar o modelo base com múltiplas ações.
Este modelo aprende padrões gerais do mercado.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.smart_trainer import SmartTrainer, DEFAULT_BASE_STOCKS
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description='Treinar modelo base com múltiplas ações'
    )
    parser.add_argument(
        '--symbols', 
        nargs='+', 
        default=None,
        help=f'Lista de símbolos (default: {DEFAULT_BASE_STOCKS[:5]}...)'
    )
    parser.add_argument(
        '--start-date', 
        type=str, 
        default='2019-01-01',
        help='Data inicial (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date', 
        type=str, 
        default=None,
        help='Data final (YYYY-MM-DD, default: hoje)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='Número de épocas'
    )
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Treinamento rápido com menos ações (5) e épocas (30)'
    )
    
    args = parser.parse_args()
    
    # Modo rápido para testes
    if args.quick:
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
        epochs = 30
        logger.info("⚡ Modo QUICK ativado!")
    else:
        symbols = args.symbols
        epochs = args.epochs
    
    logger.info("=" * 60)
    logger.info("🎓 TREINAMENTO DO MODELO BASE")
    logger.info("=" * 60)
    
    try:
        trainer = SmartTrainer(epochs=epochs)
        metrics = trainer.train_base_model(
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            epochs=epochs
        )
        
        logger.info("=" * 60)
        logger.info("✅ TREINAMENTO CONCLUÍDO!")
        logger.info("=" * 60)
        logger.info("📊 Métricas Finais:")
        logger.info(f"   RMSE: ${metrics['rmse']:.2f}")
        logger.info(f"   MAE:  ${metrics['mae']:.2f}")
        logger.info(f"   MAPE: {metrics['mape']:.2f}%")
        logger.info(f"   R²:   {metrics['r2']:.4f}")
        logger.info(f"   Dir:  {metrics['directional_accuracy']:.2f}%")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Erro no treinamento: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

