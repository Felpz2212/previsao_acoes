#!/usr/bin/env python3
"""
Script de migração para criar tabelas no PostgreSQL.

Uso:
    # Via Railway CLI (recomendado)
    railway run python -m database.migrate
    
    # Ou localmente com DATABASE_URL configurada
    python -m database.migrate
    
    # Ou diretamente
    python database/migrate.py
"""
import os
import sys
from pathlib import Path

# Adicionar o diretório backend ao path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from loguru import logger

from database.models import Base, StockPrice, Prediction, ModelMetrics, TrainingLog


def get_database_url() -> str:
    """Obtém URL do banco de dados."""
    url = os.getenv('DATABASE_URL')
    
    if not url:
        logger.error("❌ DATABASE_URL não configurada!")
        logger.info("Configure a variável de ambiente DATABASE_URL ou use 'railway run'")
        sys.exit(1)
    
    # Railway usa postgres:// mas SQLAlchemy precisa de postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    return url


def create_tables():
    """Cria todas as tabelas no banco de dados."""
    url = get_database_url()
    
    logger.info(f"🔗 Conectando ao banco: {url[:50]}...")
    
    engine = create_engine(url, echo=True)
    
    # Criar todas as tabelas
    logger.info("📝 Criando tabelas...")
    Base.metadata.create_all(engine)
    
    # Verificar tabelas criadas
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Listar tabelas
        result = session.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """))
        
        tables = [row[0] for row in result]
        
        logger.info("✅ Tabelas no banco:")
        for table in tables:
            logger.info(f"   - {table}")
        
        # Verificar contagem de registros
        for table in ['stock_prices', 'predictions', 'model_metrics', 'training_logs']:
            if table in tables:
                count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                logger.info(f"   📊 {table}: {count} registros")
        
        logger.info("✅ Migração concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ Erro durante migração: {e}")
        raise
    finally:
        session.close()


def drop_tables():
    """Remove todas as tabelas (CUIDADO!)."""
    url = get_database_url()
    
    logger.warning("⚠️ ATENÇÃO: Isso vai APAGAR todas as tabelas!")
    confirm = input("Digite 'CONFIRMAR' para continuar: ")
    
    if confirm != 'CONFIRMAR':
        logger.info("Operação cancelada.")
        return
    
    engine = create_engine(url)
    Base.metadata.drop_all(engine)
    logger.info("🗑️ Tabelas removidas.")


def show_status():
    """Mostra status do banco de dados."""
    url = get_database_url()
    
    logger.info(f"🔗 Conectando ao banco...")
    
    engine = create_engine(url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Versão do PostgreSQL
        version = session.execute(text("SELECT version()")).scalar()
        logger.info(f"📦 PostgreSQL: {version[:50]}...")
        
        # Listar tabelas
        result = session.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """))
        
        tables = [row[0] for row in result]
        
        if tables:
            logger.info(f"📋 Tabelas encontradas: {len(tables)}")
            for table in tables:
                count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                logger.info(f"   - {table}: {count} registros")
        else:
            logger.warning("⚠️ Nenhuma tabela encontrada!")
            logger.info("Execute 'python -m database.migrate create' para criar as tabelas")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        raise
    finally:
        session.close()


def main():
    """Ponto de entrada principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gerenciar banco de dados PostgreSQL")
    parser.add_argument(
        'command',
        choices=['create', 'drop', 'status'],
        default='create',
        nargs='?',
        help="Comando a executar (default: create)"
    )
    
    args = parser.parse_args()
    
    logger.info("🗃️ Database Migration Tool")
    logger.info("=" * 50)
    
    if args.command == 'create':
        create_tables()
    elif args.command == 'drop':
        drop_tables()
    elif args.command == 'status':
        show_status()


if __name__ == "__main__":
    main()

