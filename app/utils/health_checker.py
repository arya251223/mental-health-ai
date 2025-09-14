from typing import Dict, Any
from datetime import datetime
import psutil
import os
from loguru import logger

from app.services.integrated_mental_health import integrated_service
from app.core.database import engine

class SystemHealthChecker:
    """Monitor system health and AI service status"""
    
    def __init__(self):
        self.start_time = datetime.now()
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        try:
            # System metrics
            system_health = self._get_system_metrics()
            
            # Database health
            db_health = self._check_database_health()
            
            # AI service health
            ai_health = self._check_ai_services()
            
            # Overall status
            overall_status = self._calculate_overall_status(
                system_health, db_health, ai_health
            )
            
            return {
                "overall_status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "system": system_health,
                "database": db_health,
                "ai_services": ai_health,
                "emergency_resources": {
                    "crisis_lifeline": "988",
                    "crisis_text": "Text HOME to 741741",
                    "emergency": "911"
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "status": "healthy",
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "available_memory_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "available_disk_gb": round(disk.free / (1024**3), 2)
            }
            
        except Exception as e:
            logger.warning(f"System metrics collection failed: {str(e)}")
            return {
                "status": "unknown",
                "error": str(e)
            }
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and health"""
        
        try:
            # Test database connection
            with engine.connect() as connection:
                result = connection.execute("SELECT 1")
                result.fetchone()
            
            # Check database file size (for SQLite)
            db_path = "./mental_health.db"
            if os.path.exists(db_path):
                db_size_mb = round(os.path.getsize(db_path) / (1024**2), 2)
            else:
                db_size_mb = 0
            
            return {
                "status": "healthy",
                "connection": "active",
                "database_size_mb": db_size_mb
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "connection": "failed",
                "error": str(e)
            }
    
    def _check_ai_services(self) -> Dict[str, Any]:
        """Check AI services health and availability"""
        
        try:
            # Get service status
            service_status = integrated_service.get_service_status()
            
            # Count active capabilities
            capabilities = service_status.get('capabilities', {})
            active_count = sum(1 for active in capabilities.values() if active)
            total_count = len(capabilities)
            
            # Count active AI models
            ai_models = service_status.get('ai_models', {})
            active_models = sum(1 for active in ai_models.values() if active)
            total_models = len(ai_models)
            
            status = "healthy" if service_status.get('initialized') else "initializing"
            
            return {
                "status": status,
                "initialized": service_status.get('initialized', False),
                "active_capabilities": f"{active_count}/{total_count}",
                "active_models": f"{active_models}/{total_models}",
                "capabilities": capabilities,
                "models": ai_models
            }
            
        except Exception as e:
            logger.error(f"AI services health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _calculate_overall_status(
        self, 
        system_health: Dict, 
        db_health: Dict, 
        ai_health: Dict
    ) -> str:
        """Calculate overall system status"""
        
        # Critical components must be healthy
        if db_health.get("status") != "healthy":
            return "critical"
        
        # AI services can be initializing
        ai_status = ai_health.get("status")
        if ai_status not in ["healthy", "initializing"]:
            return "degraded"
        
        # System metrics are less critical
        if system_health.get("status") != "healthy":
            return "degraded"
        
        # All systems operational
        if ai_status == "initializing":
            return "starting"
        
        return "healthy"

# Global health checker instance
health_checker = SystemHealthChecker()
