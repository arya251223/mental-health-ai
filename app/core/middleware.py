from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time
from loguru import logger
import json
import uuid

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Start time
        start_time = time.time()
        
        # Get client info
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log incoming request
        logger.info(
            f"[{request_id}] Request: {request.method} {request.url.path} "
            f"from {client_host} | UA: {user_agent[:50]}..."
        )
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate process time
            process_time = time.time() - start_time
            
            # Log successful response
            logger.info(
                f"[{request_id}] Response: {response.status_code} "
                f"for {request.method} {request.url.path} "
                f"in {process_time:.4f}s"
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Calculate process time for error case
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"[{request_id}] Error: {str(e)} "
                f"for {request.method} {request.url.path} "
                f"in {process_time:.4f}s"
            )
            
            raise

class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and rate limiting"""
    
    def __init__(self, app, rate_limit_requests=100, rate_limit_window=60):
        super().__init__(app)
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self.request_counts = {}  # Simple in-memory rate limiting
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Basic rate limiting (in production, use Redis)
        client_ip = request.client.host if request.client else "unknown"
        
        if client_ip != "unknown":
            now = time.time()
            
            # Clean old entries
            self.request_counts = {
                ip: [(timestamp, count) for timestamp, count in requests 
                     if now - timestamp < self.rate_limit_window]
                for ip, requests in self.request_counts.items()
            }
            
            # Check rate limit
            if client_ip in self.request_counts:
                recent_requests = len(self.request_counts[client_ip])
                if recent_requests >= self.rate_limit_requests:
                    return Response(
                        content=json.dumps({
                            "success": False,
                            "message": "Rate limit exceeded. Please try again later.",
                            "error_code": "RATE_LIMIT_EXCEEDED"
                        }),
                        status_code=429,
                        headers={"Content-Type": "application/json"}
                    )
                
                # Add current request
                self.request_counts[client_ip].append((now, 1))
            else:
                self.request_counts[client_ip] = [(now, 1)]
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "X-Rate-Limit": f"{self.rate_limit_requests}/{self.rate_limit_window}s"
        })
        
        return response

class MentalHealthMiddleware(BaseHTTPMiddleware):
    """Specialized middleware for mental health endpoints"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if this is a mental health related endpoint
        is_mental_health_endpoint = any(
            path in str(request.url.path) 
            for path in ["/mental-health", "/chat", "/assess"]
        )
        
        if is_mental_health_endpoint:
            # Add special headers for mental health endpoints
            response = await call_next(request)
            
            response.headers.update({
                "X-Mental-Health-Support": "988 - Crisis Lifeline",
                "X-Emergency-Contact": "911 for immediate danger",
                "X-Support-Text": "Text HOME to 741741",
                "X-Disclaimer": "This service provides support but is not a replacement for professional care"
            })
            
            return response
        
        return await call_next(request)
