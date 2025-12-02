-----

# GDP Time Series Forecasting: BigQuery ML vs. Chronos-2

## Project Overview

This project explores time series forecasting for **"GDP (current US$)"** using a hybrid approach. The objective was to compare the performance of traditional statistical models against modern pre-trained foundation models when handling high-dimensional data.

The dataset includes historical GDP data alongside **48 other economic indicators**.

## Key Findings

The core insight from this experiment is that model performance is highly dependent on data complexity:

1.  **Univariate Forecasting:** `ARIMA_XREG` performed exceptionally well when looking at simple historical trends.
2.  **Multivariate Forecasting:** `Chronos-2` significantly outperformed statistical models when all **48 external indicators** were utilized, successfully capturing complex, non-linear relationships between the indicators and GDP.

## 🛠️ Tech Stack & Architecture

This project utilizes a split architecture to leverage the best tools for specific tasks:

| Component | Tool / Library | Environment | Description |
| :--- | :--- | :--- | :--- |
| **Statistical Model** | `ARIMA_XREG` | **BigQuery (`ML.FORECAST`)** | Used for baseline and univariate forecasting directly within the data warehouse. |
| **Foundation Model** | `Chronos-2` | **Local Machine / Python** | Sourced via **Hugging Face**, used for complex multivariate forecasting. |
| **Data Processing** | SQL / Pandas | BigQuery / Local | Data cleaning and feature alignment. |


## Implementation Details

### 1\. The BigQuery Approach (ARIMA\_XREG)

We utilized BigQuery's built-in machine learning capabilities to minimize data movement.

**Snippet (`sql/01_bigquery_arima.sql`):**

```sql
CREATE OR REPLACE MODEL `project.dataset.gdp_arima_model`
OPTIONS(
  model_type = 'ARIMA_PLUS_XREG',
  time_series_timestamp_col = 'date',
  time_series_data_col = 'gdp_current_usd',
  auto_arima = TRUE
) AS
SELECT
  date,
  gdp_current_usd,
  indicator_1,
  -- ... (other 47 indicators)
FROM
  `project.dataset.economic_data`
```

### 2\. The Local Approach (Chronos-2)

For the multivariate deep learning task, we utilized the `chronos-t5` family of models from Hugging Face.

**Snippet (`notebooks/02_chronos_forecast.ipynb`):**

```python
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# Context includes GDP history + 48 covariates
forecast = pipeline.predict(
    context=torch.tensor(gdp_context),
    prediction_length=12,
    num_samples=20,
)
```

## Results

  * **ARIMA\_XREG:** Best suited for identifying linear trends and seasonality in the univariate scope.
  * **Chronos-2:** Superior in handling the "noise" and complex interactions of the 48 multivariate indicators.

## Getting Started

1.  **Clone the repo:**
    `git clone https://github.com/yourusername/gdp-forecasting.git`

2.  **BigQuery Setup:**
    Ensure you have a GCP project set up and run the SQL scripts in the BigQuery console.

## Contributing

Contributions, issues, and feature requests are welcome\!

