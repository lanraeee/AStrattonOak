# StrattonOak Dashboard

Interactive trading analysis dashboard built with Next.js and React.

## Features

- **Analysis Configuration Panel**
  - Stock ticker input (AAPL, NVDA, 0700.HK, etc.)
  - Analysis date picker
  - LLM provider selection
  - Deep thinking model configuration
  - Multi-analyst selection (Market, Sentiment, News, Fundamentals)

- **Results Panel**
  - Real-time analysis updates
  - Agent reports and recommendations
  - Performance metrics visualization with charts
  - Full analysis data export

## Setup

```bash
# Install dependencies
npm install

# Set up environment
export BACKEND_URL=http://localhost:8000  # Your trading backend
export NEXT_PUBLIC_API_URL=http://localhost:8000

# Run development server
npm run dev
```

The dashboard will be available at `http://localhost:3000`

## Environment Variables

- `BACKEND_URL` - URL of the StrattonOak trading backend API
- `NEXT_PUBLIC_API_URL` - Public API URL (used client-side)

## Deployment

### To Vercel

```bash
vercel
```

Set environment variables in Vercel Dashboard:
- `BACKEND_URL` → your trading backend URL

### Docker

```bash
docker build -t strattonoak-dashboard .
docker run -p 3000:3000 -e BACKEND_URL=... strattonoak-dashboard
```

## Architecture

```
dashboard/
├── pages/
│   ├── index.tsx          # Main dashboard page
│   ├── _app.tsx           # App wrapper
│   └── api/
│       └── analyze.ts     # API proxy to backend
├── components/
│   ├── AnalysisForm.tsx   # Left configuration panel
│   └── ResultsPanel.tsx   # Right results panel
├── styles/
│   └── globals.css        # Tailwind styles
└── public/                # Static assets
```

## Backend Integration

The dashboard expects a backend API at `BACKEND_URL` with endpoint:

```
POST /api/analyze
Body: {
  ticker: string
  date: string (YYYY-MM-DD)
  provider: string (anthropic, openai, google)
  deepModel: string
  analysts: string[] (market, sentiment, news, fundamentals)
}

Response: {
  ticker: string
  date: string
  recommendation: string (BUY, SELL, HOLD)
  analysts: {
    [key: string]: { summary: string }
  }
  metrics?: array
}
```

## Technologies

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **Axios** - HTTP client
