# 🧠 Mental Health AI Support System

> **Advanced AI-powered mental health assessment, crisis detection, and therapeutic support platform**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![AI Powered](https://img.shields.io/badge/AI-Powered-purple.svg)](https://github.com/yourusername/mental-health-ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🚀 Demo & Screenshots](#-demo--screenshots)
- [⚙️ Technology Stack](#️-technology-stack)
- [🛠️ Installation & Setup](#️-installation--setup)
- [📱 Usage Guide](#-usage-guide)
- [🔧 API Documentation](#-api-documentation)
- [🎨 Frontend Interface](#-frontend-interface)
- [🧪 Testing](#-testing)
- [🚀 Deployment](#-deployment)
- [📊 Project Architecture](#-project-architecture)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🆘 Crisis Resources](#-crisis-resources)

## 🎯 Overview

**Mental Health AI** is a comprehensive, AI-powered platform designed to provide immediate mental health support, risk assessment, and crisis intervention. Built with advanced natural language processing and machine learning techniques, this system offers:

- **Real-time mental health assessment** using multi-model AI analysis
- **Crisis detection and immediate intervention** protocols
- **Therapeutic conversational AI** trained in evidence-based techniques
- **Personalized recommendations** and professional resource connections
- **24/7 accessibility** with professional-grade security

> ⚠️ **Important Medical Disclaimer**: This system provides AI-powered support and educational resources but is **not a replacement** for professional mental health care. In crisis situations, please contact **988** (Suicide & Crisis Lifeline) or emergency services.

## ✨ Key Features

### 🤖 **Advanced AI Analysis**
- **Multi-model sentiment analysis** (VADER, TextBlob, custom mental health models)
- **Emotion detection** with 8+ emotion categories
- **Risk level assessment** (Low, Medium, High, Critical)
- **Mental health condition screening** (Depression, Anxiety, PTSD, ADHD, OCD, Bipolar)
- **Linguistic pattern analysis** for deeper psychological insights

### 🚨 **Crisis Detection & Intervention**
- **Real-time crisis pattern recognition** using regex and keyword analysis
- **Immediate resource provision** (988 Crisis Line, emergency contacts)
- **Severity scoring** (1-10 scale) with automated escalation
- **Safety planning assistance** and professional referral protocols

### 💬 **Therapeutic Conversational AI**
- **Context-aware conversations** with memory of user interactions
- **Evidence-based therapeutic techniques** (CBT, DBT, Mindfulness)
- **Emotion-specific empathetic responses** tailored to user's emotional state
- **Follow-up questions** and conversation deepening strategies
- **Personalized coping strategies** based on detected conditions

### 🔐 **Security & Privacy**
- **JWT-based authentication** with secure password hashing
- **HIPAA-compliant data handling** practices
- **End-to-end encryption** for sensitive conversations
- **Anonymous support options** available
- **Comprehensive audit logging** for security monitoring

### 📊 **User Management & Analytics**
- **Personalized user profiles** with mental health tracking
- **Progress monitoring** and mood trend analysis
- **Conversation history** with privacy controls
- **Risk level tracking** over time
- **Professional resource recommendations**

### 🌐 **Professional-Grade API**
- **15+ RESTful endpoints** with comprehensive functionality
- **Interactive API documentation** (Swagger/OpenAPI)
- **Real-time WebSocket support** for chat functionality
- **Background task processing** for crisis situations
- **Comprehensive error handling** and logging

## 🚀 Demo & Screenshots

### 🌐 **Live Demo**
```
🔗 API Documentation: http://your-domain.com/api/docs
🏠 Frontend Interface: http://your-domain.com/app
📊 Health Check: http://your-domain.com/health
```

### 📱 **Key Interfaces**

**API Documentation:**
![API Docs](docs/images/api-docs.png)

**Mental Health Assessment:**
```json
{
  "risk_level": "medium",
  "predicted_conditions": ["anxiety", "depression"],
  "confidence_scores": {"anxiety": 0.75, "depression": 0.68},
  "recommendations": [
    "🫁 Practice 4-7-8 breathing technique when feeling anxious",
    "👨‍⚕️ Consider scheduling appointment with mental health professional",
    "🧘 Try mindfulness or meditation apps for daily support"
  ],
  "requires_immediate_attention": false
}
```

## ⚙️ Technology Stack

### **Backend Technologies**
| Technology | Purpose | Version |
|------------|---------|---------|
| 🐍 **Python** | Core language | 3.8+ |
| ⚡ **FastAPI** | Web framework | 0.104+ |
| 🗄️ **SQLAlchemy** | Database ORM | 2.0+ |
| 🔐 **JWT** | Authentication | Latest |
| 🤖 **Transformers** | AI/ML models | 4.35+ |
| 📊 **NLTK** | Natural language processing | 3.8+ |
| 🧠 **TextBlob** | Sentiment analysis | 0.17+ |

### **AI/ML Components**
- **VADER Sentiment Analyzer** - Social media optimized sentiment analysis
- **TextBlob Sentiment** - Statistical sentiment analysis
- **Custom Mental Health Models** - Domain-specific classification
- **Pattern Recognition** - Crisis detection algorithms
- **Linguistic Analysis** - Psychological pattern detection

### **Database & Storage**
- **SQLite** (Development) - Lightweight, embedded database
- **PostgreSQL** (Production Ready) - Scalable relational database
- **In-memory caching** - Fast conversation context storage

### **Security & Authentication**
- **bcrypt** - Password hashing
- **JWT tokens** - Stateless authentication
- **CORS middleware** - Cross-origin security
- **Request validation** - Input sanitization

## 🛠️ Installation & Setup

### **Prerequisites**
```bash
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning repository)
```

### **1. Clone Repository**
```bash
git clone https://github.com/arya251223/mental-health-ai.git
cd mental-health-ai
```

### **2. Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# Download required language models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

### **4. Environment Configuration**
Create `.env` file in project root:
```env
# Security Settings
SECRET_KEY=your-super-secret-key-here-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# Database Configuration
DATABASE_URL=sqlite:///./mental_health.db

# Application Settings
DEBUG=True
APP_NAME=Mental Health AI
ALLOWED_HOSTS=["http://localhost:3000","http://localhost:8000","*"]

# AI/ML Settings
MAX_TEXT_LENGTH=512
CONFIDENCE_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/mental_health_api.log
```

### **5. Run Application**
```bash
# Start development server
uvicorn main:app --reload

# Application will be available at:
# 🌐 API: http://localhost:8000
# 📚 Docs: http://localhost:8000/api/docs
# 🏥 Health: http://localhost:8000/health
```

## 📱 Usage Guide

### **1. API Authentication**
```bash
# Register new user
curl -X POST "http://localhost:8000/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "testuser",
    "password": "securepass123",
    "first_name": "Test",
    "data_sharing_consent": true
  }'

# Login user
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepass123"
  }'
```

### **2. Mental Health Assessment**
```bash
# Perform comprehensive assessment
curl -X POST "http://localhost:8000/api/mental-health/assess" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text_input": "I have been feeling overwhelmed and anxious lately. Work stress is getting to me and I am having trouble sleeping.",
    "assessment_type": "general"
  }'
```

### **3. AI Chat Interaction**
```bash
# Send message to therapeutic AI
curl -X POST "http://localhost:8000/api/chat/message" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "I am feeling really anxious about my job interview tomorrow. Can you help me?",
    "message_type": "user"
  }'
```

### **4. Get Mental Health Resources**
```bash
# Retrieve crisis and professional resources
curl -X GET "http://localhost:8000/api/mental-health/resources" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## 🔧 API Documentation

### **Core Endpoints**

#### **Authentication** (`/api/auth/`)
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/register` | Create new user account | ❌ |
| POST | `/login` | User authentication | ❌ |
| GET | `/me` | Get current user info | ✅ |
| POST | `/logout` | User logout | ✅ |

#### **Mental Health** (`/api/mental-health/`)
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/assess` | Comprehensive mental health analysis | ✅ |
| GET | `/resources` | Crisis and professional resources | ❌ |
| GET | `/conditions` | Supported mental health conditions | ❌ |

#### **Chat Interface** (`/api/chat/`)
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/message` | Send message to AI therapist | ✅ |
| GET | `/history` | Get conversation history | ✅ |
| DELETE | `/history` | Clear conversation history | ✅ |

#### **User Management** (`/api/users/`)
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/profile` | Get user profile | ✅ |
| GET | `/stats` | Get user statistics | ✅ |

### **Interactive Documentation**
Visit `http://localhost:8000/api/docs` for full interactive API documentation with:
- **Try it out** functionality for all endpoints
- **Request/response schemas** with examples
- **Authentication integration** for protected endpoints
- **Error response documentation**

## 🎨 Frontend Interface

### **Available Interfaces**
- **📱 Demo Frontend**: `http://localhost:8000/app` - Beautiful, responsive web interface
- **📚 API Documentation**: `http://localhost:8000/api/docs` - Interactive Swagger UI
- **🏥 Health Dashboard**: `http://localhost:8000/health` - System status monitoring

### **Frontend Features**
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Real-time Chat** - Instant AI responses with typing indicators
- **Crisis Resources** - Easy access to emergency mental health support
- **Assessment Interface** - User-friendly mental health evaluation forms
- **Progress Tracking** - Visual representation of mental health journey

## 🧪 Testing

### **Manual Testing**
```bash
# Test system health
curl http://localhost:8000/health

# Test welcome endpoint
curl http://localhost:8000/

# Test crisis detection
curl -X POST "http://localhost:8000/api/mental-health/assess" \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text_input": "I feel hopeless and want to give up on everything"}'
```

### **Automated Testing** (Future Enhancement)
```bash
# Run test suite (when implemented)
pytest tests/
pytest --cov=app tests/  # With coverage
```

## 🚀 Deployment

### **Local Development**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **Production Deployment Options**

#### **1. Railway (Recommended for beginners)**
1. Create account at [railway.app](https://railway.app)
2. Connect GitHub repository
3. Deploy automatically with zero configuration

#### **2. Heroku**
```bash
# Install Heroku CLI and login
heroku create mental-health-ai-app
git push heroku main
```

#### **3. Docker Deployment**
```dockerfile
# Dockerfile (create this file)
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### **4. Cloud Platforms**
- **AWS EC2** - Virtual server deployment
- **Google Cloud Run** - Serverless container deployment
- **DigitalOcean App Platform** - Simple application deployment

### **Environment Variables for Production**
```env
SECRET_KEY=super-secure-production-key
DATABASE_URL=postgresql://user:pass@host:port/db
DEBUG=False
ALLOWED_HOSTS=["https://yourdomain.com"]
```

## 📊 Project Architecture

### **System Architecture**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI        │    │   Database      │
│   (Web/Mobile)  │◄──►│   Backend        │◄──►│   (SQLite/      │
│                 │    │                  │    │    PostgreSQL)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   AI/ML Services │
                    │   - Sentiment    │
                    │   - Crisis Det.  │
                    │   - Chat AI      │
                    └──────────────────┘
```

### **Key Components**
1. **main.py** - Integrated FastAPI application (900+ lines)
2. **advanced_mental_health.py** - Comprehensive AI analysis engine
3. **enhanced_chat.py** - Therapeutic conversational AI
4. **Database Models** - User profiles and conversation storage
5. **Security Layer** - JWT authentication and data protection

### **Data Flow**
1. **User Input** → API Endpoint → Authentication Check
2. **Text Analysis** → AI Processing → Risk Assessment  
3. **Response Generation** → Therapeutic AI → Resource Recommendations
4. **Crisis Detection** → Immediate Intervention → Professional Referral

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### **Ways to Contribute**
- 🐛 **Bug Reports** - Report issues and bugs
- ✨ **Feature Requests** - Suggest new capabilities
- 🔧 **Code Contributions** - Submit pull requests
- 📖 **Documentation** - Improve guides and docs
- 🧪 **Testing** - Help test new features
- 🌍 **Localization** - Add language support

### **Development Setup**
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit pull request with detailed description

### **Code Standards**
- Follow PEP 8 Python style guide
- Add docstrings to all functions
- Include error handling
- Write meaningful commit messages
- Add tests for new features

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License - Copyright (c) 2024 Mental Health AI Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

## 🆘 Crisis Resources

> **If you or someone you know is experiencing a mental health crisis, please reach out for immediate help:**

### **🇺🇸 United States**
- **🚨 Crisis Lifeline**: **988** (24/7 support)
- **📱 Crisis Text**: Text **HOME** to **741741**
- **🚑 Emergency**: **911**

### **🌍 International Resources**
- **🇨🇦 Canada**: 1-833-456-4566
- **🇬🇧 UK**: 116 123 (Samaritans)
- **🇦🇺 Australia**: 13 11 14 (Lifeline)
- **🇮🇳 India**: 9152987821 (AASRA)

### **🌐 Online Support**
- **Crisis Chat**: [suicidepreventionlifeline.org](https://suicidepreventionlifeline.org)
- **Mental Health Resources**: [nami.org](https://nami.org)
- **Professional Help**: [psychologytoday.com](https://psychologytoday.com)

---

## 🙏 Acknowledgments

**Special thanks to:**
- Mental health professionals who provided guidance on therapeutic approaches
- Open source AI/ML community for foundational models and techniques  
- Beta testers who provided valuable feedback on user experience
- Crisis intervention specialists who reviewed safety protocols

---

## 📞 Contact & Support

**Project Maintainer**: [Aryan Kamble](mailto:aryan04042005@gmail.com)
**GitHub Issues**: [Report a bug or request feature](https://github.com/arya251223/mental-health-ai/issues)
**Documentation**: [Full project documentation](https://github.com/arya251223/mental-health-ai/wiki)

---

<div align="center">

**🧠 Built with ❤️ for mental health awareness and support**

**Remember: You are not alone. Help is available. Your mental health matters.**

[![GitHub stars](https://img.shields.io/github/stars/arya251223/mental-health-ai?style=social)](https://github.com/yourusername/mental-health-ai)
[![GitHub forks](https://img.shields.io/github/forks/arya251223/mental-health-ai?style=social)](https://github.com/arya251223/mental-health-ai)

</div>
