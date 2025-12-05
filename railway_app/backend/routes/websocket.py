"""
WebSocket Routes - Preços em tempo real
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict
import asyncio
import json

from services.stock_service import StockService


router = APIRouter()
stock_service = StockService()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, List[str]] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = []
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
    
    def subscribe(self, websocket: WebSocket, symbol: str):
        if websocket in self.subscriptions:
            if symbol not in self.subscriptions[websocket]:
                self.subscriptions[websocket].append(symbol)
    
    def unsubscribe(self, websocket: WebSocket, symbol: str):
        if websocket in self.subscriptions:
            if symbol in self.subscriptions[websocket]:
                self.subscriptions[websocket].remove(symbol)
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_json(message)
        except:
            self.disconnect(websocket)
    
    async def broadcast(self, symbol: str, data: dict):
        for ws, symbols in list(self.subscriptions.items()):
            if symbol in symbols:
                await self.send_personal(ws, data)


manager = ConnectionManager()


@router.websocket("/prices")
async def websocket_prices(websocket: WebSocket):
    """
    WebSocket para preços em tempo real.
    
    Mensagens:
    - {"action": "subscribe", "symbol": "AAPL"}
    - {"action": "unsubscribe", "symbol": "AAPL"}
    """
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            action = message.get("action")
            symbol = message.get("symbol", "").upper()
            
            if action == "subscribe" and symbol:
                manager.subscribe(websocket, symbol)
                
                # Enviar dados iniciais
                try:
                    stock_data = stock_service.get_stock_data(symbol, days=5)
                    if stock_data is not None and len(stock_data) > 0:
                        current = float(stock_data['close'].iloc[-1])
                        prev = float(stock_data['close'].iloc[-2]) if len(stock_data) > 1 else current
                        change = ((current - prev) / prev) * 100
                        
                        await manager.send_personal(websocket, {
                            "type": "price_update",
                            "symbol": symbol,
                            "price": round(current, 2),
                            "change_percent": round(change, 2),
                            "timestamp": str(stock_data['timestamp'].iloc[-1])
                        })
                except Exception as e:
                    await manager.send_personal(websocket, {
                        "type": "error",
                        "symbol": symbol,
                        "message": str(e)
                    })
                
                await manager.send_personal(websocket, {
                    "type": "subscribed",
                    "symbol": symbol
                })
            
            elif action == "unsubscribe" and symbol:
                manager.unsubscribe(websocket, symbol)
                await manager.send_personal(websocket, {
                    "type": "unsubscribed",
                    "symbol": symbol
                })
            
            elif action == "ping":
                await manager.send_personal(websocket, {"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)


async def price_updater():
    """Background task para atualizar preços."""
    while True:
        await asyncio.sleep(30)  # Atualiza a cada 30 segundos
        
        all_symbols = set()
        for symbols in manager.subscriptions.values():
            all_symbols.update(symbols)
        
        for symbol in all_symbols:
            try:
                stock_data = stock_service.get_stock_data(symbol, days=2)
                if stock_data is not None and len(stock_data) > 0:
                    current = float(stock_data['close'].iloc[-1])
                    prev = float(stock_data['close'].iloc[-2]) if len(stock_data) > 1 else current
                    change = ((current - prev) / prev) * 100
                    
                    await manager.broadcast(symbol, {
                        "type": "price_update",
                        "symbol": symbol,
                        "price": round(current, 2),
                        "change_percent": round(change, 2)
                    })
            except:
                pass

