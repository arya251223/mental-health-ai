from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from loguru import logger
import traceback
from datetime import datetime

class MentalHealthException(Exception):
    """Base exception for mental health specific errors"""
    
    def __init__(self, message: str, error_code: str = "MENTAL_HEALTH_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class CrisisDetectedException(MentalHealthException):
    """Exception raised when crisis situation is detected"""
    
    def __init__(self, message: str = "Crisis situation detected", severity: int = 5):
        self.severity = severity
        super().__init__(message, "CRISIS_DETECTED")

class AssessmentException(MentalHealthException):
    """Exception for assessment processing errors"""
    
    def __init__(self, message: str = "Assessment processing failed"):
        super().__init__(message, "ASSESSMENT_ERROR")

async def crisis_exception_handler(request: Request, exc: CrisisDetectedException):
    """Special handler for crisis situations - highest priority"""
    request_id = getattr(request.state, "request_id", "unknown")
    client_ip = request.client.host if request.client else "unknown"
    
    logger.critical(
        f"[{request_id}] CRISIS DETECTED: {exc.message} | "
        f"Severity: {exc.severity}/10 | IP: {client_ip}"
    )
    
    return JSONResponse(
        status_code=200,  # Don't use error status for crisis response
        content={
            "success": True,
            "message": "We're here to help you through this difficult time.",
            "crisis_detected": True,
            "severity_level": exc.severity,
            "immediate_resources": {
                "crisis_lifeline": {
                    "number": "988",
                    "description": "24/7 Suicide & Crisis Lifeline",
                    "action": "Call or text 988"
                },
                "crisis_text": {
                    "number": "741741", 
                    "description": "Crisis Text Line",
                    "action": "Text HOME to 741741"
                },
                "emergency": {
                    "number": "911",
                    "description": "Emergency Services",
                    "when": "If you're in immediate physical danger"
                }
            },
            "immediate_steps": [
                "You are not alone in this - help is available",
                "Please contact one of the crisis resources above immediately",
                "Consider reaching out to a trusted friend, family member, or counselor",
                "If possible, try to stay with someone or have someone come to you",
                "Remove any means of self-harm from your immediate area"
            ],
            "followup_resources": [
                "Schedule an appointment with a mental health professional",
                "Contact your primary care doctor",
                "Look into local support groups",
                "Consider intensive outpatient programs if needed"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def mental_health_exception_handler(request: Request, exc: MentalHealthException):
    """Handler for general mental health exceptions"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(f"[{request_id}] Mental Health Exception: {exc.error_code} - {exc.message}")
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": exc.message,
            "error_code": exc.error_code,
            "support_available": True,
            "resources": {
                "crisis_line": "Call or text 988",
                "text_support": "Text HOME to 741741",
                "emergency": "Call 911 if in immediate danger"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler for request validation errors with helpful messages"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(f"[{request_id}] Validation error for {request.url.path}: {exc.errors()}")
    
    # Make validation errors more user-friendly
    friendly_errors = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        message = error["msg"]
        friendly_errors.append(f"{field}: {message}")
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Please check your input and try again",
            "errors": friendly_errors,
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handler for HTTP exceptions with consistent format"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(f"[{request_id}] HTTP {exc.status_code} for {request.url.path}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handler for unexpected exceptions"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"[{request_id}] Unhandled exception for {request.url.path}: {str(exc)}\n"
        f"Traceback: {traceback.format_exc()}"
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected error occurred. Our team has been notified.",
            "error_type": type(exc).__name__,
            "request_id": request_id,
            "support": "If this persists, please contact support",
            "timestamp": datetime.utcnow().isoformat()
        }
    )