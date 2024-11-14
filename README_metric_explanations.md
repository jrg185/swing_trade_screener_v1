# swing_trade_screener_v1
First attempt at building an ensemble ML model to predict profitable swing trades based on market data pulled via API

# Stock Screening Metrics Documentation

## 1. Model Quality Metrics

### R-Squared (R²)
- **Description**: Measures how well the model explains price movements
- **Range**: 0 to 1 (0% to 100%)
- **Calculation**: `r2_score(actual_values, predicted_values)`
- **Threshold**: > 0.4 (40%)
- **Interpretation**: Higher is better; indicates model's explanatory power

### Trade Success Ratio
- **Description**: Proportion of successful trades in backtest
- **Calculation**: `profitable_trades / total_trades`
- **Example**: 0.75 means 75% of trades were profitable
- **Components**:
  - Total Trades: Number of trades in backtest period
  - Profitable Trades: Number of correctly predicted moves

## 2. Prediction Metrics

### Direction
- **Description**: Predicted price movement
- **Values**: 'Up' or 'Down'
- **Calculation**: Based on prediction confidence
  - Up: confidence > 0.5
  - Down: confidence < 0.5

### Confidence
- **Description**: Raw probability from model
- **Range**: 0 to 1
- **Interpretation**: 
  - > 0.5: Upward move predicted
  - < 0.5: Downward move predicted
  - Further from 0.5 = stronger signal

### Conviction
- **Description**: Strength of signal normalized to 0-1 scale
- **Calculation**: `abs(confidence - 0.5) * 2`
- **Example**: 
  - Confidence of 0.8 → Conviction = 0.6
  - Confidence of 0.2 → Conviction = 0.6

### Win Rate
- **Description**: Overall prediction accuracy
- **Calculation**: `correct_predictions / total_predictions`
- **Threshold**: > 0.6 (60%)
- **Interpretation**: Higher indicates more reliable predictions

### Precision
- **Description**: Accuracy for specific direction predicted
- **Calculation**: Separate for up/down predictions
  - `correct_up_predictions / total_up_predictions`
  - `correct_down_predictions / total_down_predictions`
- **Threshold**: > 0.6 (60%)

## 3. Price Target Metrics

### Expected Move
- **Description**: Predicted price movement over forecast period
- **Calculation**: 
  ```python
  volatility * confidence / sqrt(252) * forecast_days
  ```
- **Components**:
  - Volatility: Annualized standard deviation of returns
  - Confidence: Model's prediction probability
  - Forecast period: 5 days by default

### Target Price
- **Description**: Price objective based on expected move
- **Calculation**:
  ```python
  if direction == 'Up':
      current_price * (1 + expected_move_pct)
  else:
      current_price * (1 - expected_move_pct)
  ```

### Stop Loss
- **Description**: Exit price for risk management
- **Calculation**:
  ```python
  if direction == 'Up':
      current_price * (1 - expected_move_pct * 0.5)
  else:
      current_price * (1 + expected_move_pct * 0.5)
  ```
- **Note**: Set at half the expected move in opposite direction

### Support/Resistance Levels
- **Description**: Technical price boundaries
- **Calculation**:
  ```python
  support = current_price - (1.5 * ATR)
  resistance = current_price + (1.5 * ATR)
  ```
- **Component**: ATR (Average True Range) measures volatility

## 4. Risk Metrics

### Average Win
- **Description**: Mean return on successful trades
- **Calculation**: Average of positive returns in backtest
- **Interpretation**: Higher indicates more profitable winners

### Average Loss
- **Description**: Mean return on unsuccessful trades
- **Calculation**: Average of negative returns in backtest
- **Interpretation**: Closer to zero indicates better loss control

### ATR (Average True Range)
- **Description**: Measure of price volatility
- **Calculation**: 14-period average of true range
  ```python
  true_range = max(
      high - low,
      abs(high - previous_close),
      abs(low - previous_close)
  )
  ```
- **Usage**: Used for setting price targets and stops

## 5. Technical Scores

### Total Technical Score
- **Description**: Composite of technical factors
- **Components**:
  - Trend Score (35%): Moving average relationships
  - Momentum Score (25%): RSI and MACD indicators
  - Volume Score (20%): Volume patterns and trends
  - Support/Resistance Score (20%): Price positioning
- **Range**: 0 to 100
- **Interpretation**: Higher indicates stronger technical position

## 6. Filter Criteria

Current screening thresholds:
- R² > 0.4
- Win Rate > 60%
- Precision > 60%
- Strong conviction (confidence > 0.6 or < 0.4)

## 7. Final Score Calculation

The final score combines multiple factors:
```python
score = (Conviction * 
         Win_Rate * 
         R_Squared * 
         (Technical_Score/100) * 
         (1 + Trade_Ratio) * 
         (1 + Avg_Win/(1 + Avg_Loss)))
```

This weighted approach considers:
- Signal strength (Conviction)
- Historical accuracy (Win Rate)
- Model quality (R²)
- Technical factors
- Risk/reward characteristics
