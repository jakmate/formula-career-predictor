# Formula Career Predictor

[![codecov](https://codecov.io/gh/jakmate/formula-career-predictor/branch/main/graph/badge.svg)](https://codecov.io/gh/jakmate/formula-career-predictor)
[![CI/CD](https://github.com/jakmate/formula-career-predictor/workflows/CI%2FCD/badge.svg)](https://github.com/jakmate/formula-career-predictor/actions)

A machine learning application that predicts Formula 3 drivers' likelihood of advancing to Formula 2, using historical performance data and advanced ML models.

## Tech Stack

- **FastAPI** - REST API framework
- **React** - UI framework

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

### Endpoints
- `GET /api/predictions` - Get all model predictions
- `GET /api/predictions/{model_name}` - Get specific model predictions
- `GET /api/health` - Health check with system status
- `POST /api/refresh` - Trigger data refresh
- `GET /api/races/{series}` - Get F1/F2/F3 race schedules
- `GET /api/races/{series}/next` - Get next upcoming race

## Development

### Running Tests
```bash
# Backend
cd backend
pytest --cov=. --cov-report=html

# Frontend
cd frontend
npm run test:coverage
```

### Code Quality
```bash
# Backend linting
flake8 --max-line-length=100 .

# Frontend linting
npx eslint . --ext .js,.jsx,.ts,.tsx
npx prettier --check "src/**/*.{js,jsx,ts,tsx,json,css,md}"
```

## Deployment

The application is containerized and deployed on:
- **Backend**: FastAPI on Render
- **Frontend**: Vite on Render
- **CI/CD**: GitHub Actions for testing and deployment

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Official Formula 1, Formula 2, and Formula 3 championships for race data
- Wikipedia contributors for driver and team information