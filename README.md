
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================
# 0. 參數（可調）
# =========================
period = 28             # 布林通道與MA視窗
atr_period = 14
adx_period = 14
rsi_period = 14
ctx_window = 20         # 布林帶寬度過去均值的參考視窗

# 品質過濾門檻  可最佳化
ADX_MIN = 21            # 趨勢強度
BB_EXPANSION_RATIO = 1.2  # 當前布林寬度相對過去均值的擴張比
RSI_MIN_BUY = 55        # 多單 RSI 至少偏強
RSI_MAX_SELL = 45       # 空單 RSI 至少偏弱（對應上界 45）
BREAK_ATR_MIN = 0.30    # 突破強度（以 ATR 標準化）
VOL_RATIO_MIN = 1.20    # 量能濾網（若無 volume 欄位則自動忽略）
REQUIRE_SQUEEZE_OFF = True  # 要求從壓縮轉為擴張（Keltner 嵌布林 → 擴張）

# 1. 讀取 CSV
df = pd.read_csv("BTCUSDT-4h-data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
for c in ['open','high','low','close']:
    df[c] = df[c].astype(float)

# =========================
# 2. 計算指標（含布林、ATR、ADX、RSI、Keltner、品質欄位）
# =========================
# 布林通道（母體標準差 ddof=0 + min_periods）
df['MA']   = df['close'].rolling(window=period, min_periods=period).mean()
df['STD']  = df['close'].rolling(window=period, min_periods=period).std(ddof=0)
df['Upper'] = df['MA'] + 2*df['STD']
df['Lower'] = df['MA'] - 2*df['STD']
df['BB_Width'] = df['Upper'] - df['Lower']
df['PB'] = (df['close'] - df['Lower']) / (df['Upper'] - df['Lower'])

# True Range / ATR（Wilder）
prev_close = df['close'].shift(1)
tr1 = df['high'] - df['low']
tr2 = (df['high'] - prev_close).abs()
tr3 = (df['low'] - prev_close).abs()
df['TR'] = np.maximum(tr1, np.maximum(tr2, tr3))

df['ATR'] = np.nan
if len(df) >= atr_period:
    df.loc[df.index[atr_period-1], 'ATR'] = df['TR'].iloc[:atr_period].mean()
    for i in range(atr_period, len(df)):
        df.loc[df.index[i], 'ATR'] = (df['ATR'].iloc[i-1]*(atr_period-1) + df['TR'].iloc[i]) / atr_period

# DI+/DI-/ADX（Wilder）
df['upMove'] = df['high'] - df['high'].shift(1)
df['downMove'] = df['low'].shift(1) - df['low']
df['plusDM'] = np.where((df['upMove'] > df['downMove']) & (df['upMove'] > 0), df['upMove'], 0.0)
df['minusDM'] = np.where((df['downMove'] > df['upMove']) & (df['downMove'] > 0), df['downMove'], 0.0)

df['plusDM_s'] = np.nan
df['minusDM_s'] = np.nan
df['TR_s'] = np.nan
if len(df) >= adx_period:
    df.loc[df.index[adx_period-1], 'plusDM_s'] = df['plusDM'].iloc[:adx_period].mean()
    df.loc[df.index[adx_period-1], 'minusDM_s'] = df['minusDM'].iloc[:adx_period].mean()
    df.loc[df.index[adx_period-1], 'TR_s'] = df['TR'].iloc[:adx_period].mean()
    for i in range(adx_period, len(df)):
        df.loc[df.index[i], 'plusDM_s'] = df['plusDM_s'].iloc[i-1] - (df['plusDM_s'].iloc[i-1]/adx_period) + df['plusDM'].iloc[i]
        df.loc[df.index[i], 'minusDM_s'] = df['minusDM_s'].iloc[i-1] - (df['minusDM_s'].iloc[i-1]/adx_period) + df['minusDM'].iloc[i]
        df.loc[df.index[i], 'TR_s'] = df['TR_s'].iloc[i-1] - (df['TR_s'].iloc[i-1]/adx_period) + df['TR'].iloc[i]

df['plusDI'] = 100 * (df['plusDM_s'] / df['TR_s'])
df['minusDI'] = 100 * (df['minusDM_s'] / df['TR_s'])
df['DX'] = 100 * ((df['plusDI'] - df['minusDI']).abs() / (df['plusDI'] + df['minusDI']))

df['ADX'] = np.nan
first_adx_idx = adx_period*2 - 1
if len(df) > first_adx_idx:
    df.loc[df.index[first_adx_idx], 'ADX'] = df['DX'].iloc[adx_period:first_adx_idx+1].mean()
    for i in range(first_adx_idx+1, len(df)):
        df.loc[df.index[i], 'ADX'] = (df['ADX'].iloc[i-1]*(adx_period-1) + df['DX'].iloc[i]) / adx_period

# RSI（Wilder）
delta = df['close'].diff()
gain = delta.clip(lower=0.0)
loss = -delta.clip(upper=0.0)
df['avgGain'] = np.nan
df['avgLoss'] = np.nan
if len(df) >= rsi_period+1:
    df.loc[df.index[rsi_period], 'avgGain'] = gain.iloc[1:rsi_period+1].mean()
    df.loc[df.index[rsi_period], 'avgLoss'] = loss.iloc[1:rsi_period+1].mean()
    for i in range(rsi_period+1, len(df)):
        df.loc[df.index[i], 'avgGain'] = (df['avgGain'].iloc[i-1]*(rsi_period-1) + gain.iloc[i]) / rsi_period
        df.loc[df.index[i], 'avgLoss'] = (df['avgLoss'].iloc[i-1]*(rsi_period-1) + loss.iloc[i]) / rsi_period
rs = df['avgGain'] / df['avgLoss']
df['RSI'] = 100 - (100 / (1 + rs))

# Keltner Channel（基於 EMA20 ± 1.5*ATR）
df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
df['KC_Upper'] = df['EMA20'] + 1.5*df['ATR']
df['KC_Lower'] = df['EMA20'] - 1.5*df['ATR']
df['KC_Width'] = df['KC_Upper'] - df['KC_Lower']
df['SqueezeOn']  = (df['BB_Width'] < df['KC_Width'])   # 布林在肯特納內 → 壓縮
df['SqueezeOff'] = (df['BB_Width'] > df['KC_Width'])   # 擴張

# 量能濾網（若有 volume 欄位）
if 'volume' in df.columns:
    df['Vol_MA'] = df['volume'].rolling(20, min_periods=20).mean()
    df['Vol_Ratio'] = df['volume'] / df['Vol_MA']

# 突破強度（以 ATR 標準化）
df['BreakUpStrength_ATR']   = (df['close'] - df['Upper']) / df['ATR']     # 多：收盤相對上軌
df['BreakDownStrength_ATR'] = (df['Lower'] - df['close']) / df['ATR']     # 空：收盤相對下軌

# 布林帶寬度的過去均值（品質過濾用）
df['BB_Width_MA'] = df['BB_Width'].rolling(ctx_window, min_periods=ctx_window).mean()

# =========================
# 3. 品質過濾函式
# =========================
def quality_ok(row, side='BUY'):
    # ADX
    adx = row['ADX']
    if np.isnan(adx) or adx < ADX_MIN:
        return False

    # 布林帶擴張相對過去均值
    bb_w = row['BB_Width']
    bb_ma = row['BB_Width_MA']
    if np.isnan(bb_w) or np.isnan(bb_ma) or bb_w < bb_ma * BB_EXPANSION_RATIO:
        return False

    # RSI
    rsi = row['RSI']
    if np.isnan(rsi):
        return False
    if side == 'BUY' and rsi < RSI_MIN_BUY:
        return False
    if side == 'SELL' and rsi > RSI_MAX_SELL:
        return False

    # Keltner 擴張
    if REQUIRE_SQUEEZE_OFF and not bool(row['SqueezeOff']):
        return False

    # 突破強度（ATR 標準化）
    if side == 'BUY':
        bs = row['BreakUpStrength_ATR']
    else:
        bs = row['BreakDownStrength_ATR']
    if np.isnan(bs) or bs < BREAK_ATR_MIN:
        return False

    # 量能濾網（若有）
    if 'Vol_Ratio' in row.index:
        vr = row['Vol_Ratio']
        if np.isnan(vr) or vr < VOL_RATIO_MIN:
            return False

    return True

# =========================
# 4. 回測邏輯（加入品質過濾）
# =========================
initial_balance = 10000
balance = initial_balance          # 保持為初始資金 + 累積已平倉 PnL
position = 0
entry_price = None
take_profit = None
stop_loss = None
trade_log = []

for i in range(period, len(df)):
    row = df.iloc[i]
    price = row['close']
    upper = row['Upper']
    lower = row['Lower']
    middle = row['MA']
    high = row['high']
    low = row['low']
    ts = row['timestamp']

    if position == 0:
        # 多單：收盤突破上軌 + 品質通過
        if (price > upper) and quality_ok(row, side='BUY'):
            position = 0.3
            entry_price = price
            take_profit = entry_price + 1500
            stop_loss = middle - 300
            trade_log.append({
                "EntryTime": ts, "Type": "BUY", "EntryPrice": entry_price,
                "ADX": row['ADX'], "RSI": row['RSI'],
                "BB_Width": row['BB_Width'], "BB_Width_MA": row['BB_Width_MA'],
                "BreakStrength_ATR": row['BreakUpStrength_ATR'],
                "SqueezeOff": bool(row['SqueezeOff']),
                "Vol_Ratio": row['Vol_Ratio'] if 'Vol_Ratio' in df.columns else None
            })

        # 空單：收盤跌破下軌 + 品質通過
        elif (price < lower) and quality_ok(row, side='SELL'):
            position = 0.3
            entry_price = price
            take_profit = entry_price - 2400
            stop_loss = middle
            trade_log.append({
                "EntryTime": ts, "Type": "SELL", "EntryPrice": entry_price,
                "ADX": row['ADX'], "RSI": row['RSI'],
                "BB_Width": row['BB_Width'], "BB_Width_MA": row['BB_Width_MA'],
                "BreakStrength_ATR": row['BreakDownStrength_ATR'],
                "SqueezeOff": bool(row['SqueezeOff']),
                "Vol_Ratio": row['Vol_Ratio'] if 'Vol_Ratio' in df.columns else None
            })
    else:
        # 出場邏輯沿用（intrabar 觸價）
        if trade_log[-1]['Type'] == "BUY":
            if high >= take_profit:
                exit_price = take_profit
                pnl = (exit_price - entry_price) * position
                balance += pnl
                trade_log[-1].update({"ExitTime": ts, "ExitPrice": exit_price, "PnL": pnl})
                position = 0
            elif low <= stop_loss:
                exit_price = stop_loss
                pnl = (exit_price - entry_price) * position
                balance += pnl
                trade_log[-1].update({"ExitTime": ts, "ExitPrice": exit_price, "PnL": pnl})
                position = 0
        else:  # SELL
            if low <= take_profit:
                exit_price = take_profit
                pnl = (entry_price - exit_price) * position
                balance += pnl
                trade_log[-1].update({"ExitTime": ts, "ExitPrice": exit_price, "PnL": pnl})
                position = 0
            elif high >= stop_loss:
                exit_price = stop_loss
                pnl = (entry_price - exit_price) * position
                balance += pnl
                trade_log[-1].update({"ExitTime": ts, "ExitPrice": exit_price, "PnL": pnl})
                position = 0

# 若最後仍未平倉，可選擇用最後收盤價市值化（可開關）
mark_to_market = False
if mark_to_market and position != 0:
    last_price = df['close'].iloc[-1]
    if trade_log[-1]['Type'] == "BUY":
        pnl = (last_price - entry_price) * position
    else:
        pnl = (entry_price - last_price) * position
    balance += pnl
    trade_log[-1].update({"ExitTime": df['timestamp'].iloc[-1], "ExitPrice": last_price, "PnL": pnl})
    position = 0

# =========================
# 5. 計算績效指標
# =========================
trades_df = pd.DataFrame(trade_log)
final_balance = initial_balance + trades_df['PnL'].dropna().sum()

win_rate = (trades_df['PnL'] > 0).mean() * 100 if not trades_df.empty else 0

# 資金曲線（每次平倉後）
balances_curve = [initial_balance]
x_curve = [pd.Timestamp(df['timestamp'].iloc[period])]
current_balance = initial_balance
for _, t in trades_df.iterrows():
    if pd.notna(t.get('PnL', np.nan)):
        current_balance += t['PnL']
        x_curve.append(t['ExitTime'])
        balances_curve.append(current_balance)

# 最大回撤
peak = balances_curve[0]
max_drawdown = 0
for b in balances_curve:
    if b > peak:
        peak = b
    drawdown = (peak - b) / peak
    if drawdown > max_drawdown:
        max_drawdown = drawdown

# Sharpe（以每筆交易回報粗估，注意年化不精準）
returns = trades_df['PnL'] / initial_balance if not trades_df.empty else pd.Series(dtype=float)
sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) != 0 else 0

# 6. 輸出結果
print("===== 策略績效報告 =====")
print(f"初始資金: {initial_balance}")
print(f"最終資金: {final_balance}")
print(f"報酬率: {(final_balance - initial_balance)/initial_balance*100:.2f}%")
print(f"勝率: {win_rate:.2f}%")
print(f"最大回撤: {max_drawdown*100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print("===== 交易紀錄（含品質指標）=====")
print(trades_df)

# 7. 資金曲線圖（按平倉時間對齊）
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_curve, y=balances_curve, mode='lines+markers', name='Equity Curve'))
fig.update_layout(title='策略資金曲線', xaxis_title='時間', yaxis_title='資金')
fig.show()
