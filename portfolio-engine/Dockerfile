# One image for the API and the dashboard.
#   docker build -t portfolio-engine .
#   docker run -p 8000:8000 portfolio-engine                      # API at http://localhost:8000/docs
#   docker run -p 8501:8501 portfolio-engine streamlit run app/streamlit_app.py --server.address=0.0.0.0
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[product]"
COPY configs ./configs
COPY data ./data
COPY experiments ./experiments
COPY app ./app
EXPOSE 8000 8501
CMD ["uvicorn", "portfolio_engine.product.api:app", "--host", "0.0.0.0", "--port", "8000"]
