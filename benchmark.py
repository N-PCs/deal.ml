import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.rag_engine.hybrid_retriever import FinancialHybridRetriever
from src.ml_engine.metrics_forecaster import FinancialMetricsForecaster

# 1. RAG Benchmarking
print("--- RAG Benchmarks ---")
retriever = FinancialHybridRetriever()
docs = [
    {"content": "Apple Q3 revenue grew by 5% year over year.", "metadata": {"source": "AAPL_Q3"}},
    {"content": "Microsoft reported a 10% increase in Azure cloud revenue.", "metadata": {"source": "MSFT_Q3"}},
    {"content": "Tesla deliveries fell short of analyst expectations in Q2.", "metadata": {"source": "TSLA_Q2"}},
    {"content": "Nvidia datacenter sales skyrocketed due to AI demand.", "metadata": {"source": "NVDA_Q1"}},
    {"content": "Amazon AWS margins improved significantly.", "metadata": {"source": "AMZN_Q2"}}
]
start_time = time.time()
retriever.index_documents(docs)
index_time = time.time() - start_time
print(f"RAG Indexing Time: {index_time:.4f} seconds")

start_time = time.time()
results = retriever.hybrid_search("cloud computing growth", top_k=2)
search_time = time.time() - start_time
print(f"RAG Search Time: {search_time:.4f} seconds")
print(f"RAG Top Result Score: {results[0]['score']}")

# 2. ML Regressor Benchmarking
print("\n--- ML Regressor Benchmarks ---")
ml_engine = FinancialMetricsForecaster()
X_train, y_train = ml_engine.generate_synthetic_training_data(n_samples=800)
X_test, y_test = ml_engine.generate_synthetic_training_data(n_samples=200)

ml_engine.train_model()
y_pred = ml_engine.model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R2 Score: {r2:.4f}")

