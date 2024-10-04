import asyncio
import aiohttp
import pandas as pd
import numpy as np
import discord
from discord.ext import commands
from datetime import datetime, time, timedelta
import pytz
import time

FMP_API_KEY = 'temp'
DISCORD_TOKEN = 'temp'
DISCORD_CHANNEL_ID = 1

CHECK_INTERVAL = 1  # Check every second
ALERT_COOLDOWN = 15 * 60  # 15 minutes in seconds
SIGNIFICANT_CHANGE_THRESHOLD = 3.0  # 3% change threshold
PRICE_CHANGE_THRESHOLD = 0.5
VOLUME_CHANGE_THRESHOLD = 100
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
BOLLINGER_BAND_THRESHOLD = 2.0
BOLLINGER_BAND_ROC_THRESHOLD = 0.5
EMA_CROSS_THRESHOLD = 0.1
MACD_SIGNAL_THRESHOLD = 0.1
EMA200_DEVIATION_THRESHOLD = 2

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

user_tracking = {}
symbol_subscribers = {}
last_alert_times = {}
last_alert_prices = {}

def is_market_open():
    pst_tz = pytz.timezone('US/Pacific')
    now = datetime.now(pst_tz)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    market_open = now.replace(hour=6, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)
    return market_open <= now < market_close

async def get_fmp_data(session, endpoint, symbol):
    url = f'https://financialmodelingprep.com/api/v3/{endpoint}/{symbol}?apikey={FMP_API_KEY}'
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        print(f"Error fetching data for {symbol}: {await response.text()}")
        return None

def calculate_indicators(df):
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['STD20'] = df['close'].rolling(window=20).std()
    df['Upper_BB'] = df['SMA20'] + (df['STD20'] * 2)
    df['Lower_BB'] = df['SMA20'] - (df['STD20'] * 2)
    df['Upper_BB_ROC'] = df['Upper_BB'].pct_change() * 100
    df['Lower_BB_ROC'] = df['Lower_BB'].pct_change() * 100
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

async def check_indicators(session, symbol):
    quote_data = await get_fmp_data(session, 'quote', symbol)
    if not quote_data:
        return None

    historical_data = await get_fmp_data(session, 'historical-price-full', symbol)
    if not historical_data or 'historical' not in historical_data:
        return None

    df = pd.DataFrame(historical_data['historical'][:200][::-1])
    df = calculate_indicators(df)

    current_price = quote_data[0]['price']
    print(f"Checking ${symbol} - Current Price: ${current_price:.2f}")
    current_volume = quote_data[0]['volume']
    last_close = df['close'].iloc[-1]
    last_volume = df['volume'].iloc[-1]

    current_time = time.time()
    last_alert_time = last_alert_times.get(symbol, 0)
    last_alert_price = last_alert_prices.get(symbol, current_price)

    time_since_last_alert = current_time - last_alert_time
    price_change_since_last_alert = abs((current_price - last_alert_price) / last_alert_price * 100)

    should_alert = (time_since_last_alert >= ALERT_COOLDOWN or 
                    price_change_since_last_alert >= SIGNIFICANT_CHANGE_THRESHOLD)

    if should_alert:
        alerts = []

        price_change_percent = (current_price - last_close) / last_close * 100
        if abs(price_change_percent) >= PRICE_CHANGE_THRESHOLD:
            alerts.append(f"**Price** {'increased' if price_change_percent > 0 else 'decreased'} by {abs(price_change_percent):.2f}%")

        if last_volume > 0:
            volume_change_percent = (current_volume - last_volume) / last_volume * 100
            if volume_change_percent >= VOLUME_CHANGE_THRESHOLD:
                alerts.append(f"**Volume** spiked by {volume_change_percent:.2f}%")

        current_rsi = df['RSI'].iloc[-1]
        if current_rsi >= RSI_OVERBOUGHT:
            alerts.append(f"**RSI** is overbought at {current_rsi:.2f} (threshold: {RSI_OVERBOUGHT})")
        elif current_rsi <= RSI_OVERSOLD:
            alerts.append(f"**RSI** is oversold at {current_rsi:.2f} (threshold: {RSI_OVERSOLD})")

        upper_bb = df['Upper_BB'].iloc[-1]
        lower_bb = df['Lower_BB'].iloc[-1]
        if current_price >= upper_bb * (1 + BOLLINGER_BAND_THRESHOLD / 100):
            alerts.append(f"**Price** (${current_price:.2f}) is above upper Bollinger Band (${upper_bb:.2f}) by {((current_price / upper_bb - 1) * 100):.2f}%")
        elif current_price <= lower_bb * (1 - BOLLINGER_BAND_THRESHOLD / 100):
            alerts.append(f"**Price** (${current_price:.2f}) is below lower Bollinger Band (${lower_bb:.2f}) by {((lower_bb / current_price - 1) * 100):.2f}%")

        ema9 = df['EMA9'].iloc[-1]
        ema20 = df['EMA20'].iloc[-1]
        ema50 = df['EMA50'].iloc[-1]
        ema200 = df['EMA200'].iloc[-1]

        if abs(ema9 - ema20) / ema20 * 100 <= EMA_CROSS_THRESHOLD:
            alerts.append(f"**EMA9** (${ema9:.2f}) is crossing EMA20 (${ema20:.2f})")
        if abs(ema20 - ema50) / ema50 * 100 <= EMA_CROSS_THRESHOLD:
            alerts.append(f"**EMA20** (${ema20:.2f}) is crossing EMA50 (${ema50:.2f})")

        ema200_deviation = (current_price - ema200) / ema200 * 100
        if abs(ema200_deviation) >= EMA200_DEVIATION_THRESHOLD:
            alerts.append(f"**Price** (${current_price:.2f}) has deviated from EMA200 (${ema200:.2f}) by {ema200_deviation:.2f}%")

        macd = df['MACD'].iloc[-1]
        signal_line = df['Signal_Line'].iloc[-1]
        if abs(macd - signal_line) >= current_price * MACD_SIGNAL_THRESHOLD / 100:
            alerts.append(f"**MACD** ({macd:.4f}) is {'bullish' if macd > signal_line else 'bearish'} compared to Signal Line ({signal_line:.4f})")

        if alerts:
            last_alert_times[symbol] = current_time
            last_alert_prices[symbol] = current_price
            importance = "**IMPORTANT** " if len(alerts) > 1 else ""
            message = f"{importance}ALERT for {symbol}:\n" + "\n".join(alerts)
            return message

    return None

async def send_channel_message(message, symbol):
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if channel:
        try:
            subscribers = symbol_subscribers.get(symbol, set())
            mentions = ' '.join(f'<@{user_id}>' for user_id in subscribers)
            full_message = f"{mentions}\n{message}"
            await channel.send(full_message)
        except discord.Forbidden:
            print(f"Cannot send message to channel ID {DISCORD_CHANNEL_ID}. Check permissions.")
    else:
        print(f"Channel with ID {DISCORD_CHANNEL_ID} not found.")

async def monitor_stocks():
    await bot.wait_until_ready()
    print("Stock monitoring task started")
    async with aiohttp.ClientSession() as session:
        while not bot.is_closed():
            if is_market_open():
                for symbol in symbol_subscribers.keys():
                    message = await check_indicators(session, symbol)
                    if message:
                        await send_channel_message(message, symbol)
                await asyncio.sleep(CHECK_INTERVAL)
            else:
                next_open = get_next_market_open(datetime.now(pytz.timezone('US/Pacific')))
                sleep_duration = (next_open - datetime.now(pytz.timezone('US/Pacific'))).total_seconds()
                print(f"Market is closed. Sleeping until {next_open.strftime('%Y-%m-%d %I:%M %p PST')}")
                await asyncio.sleep(sleep_duration)

def get_next_market_open(current_time):
    pst_tz = pytz.timezone('US/Pacific')
    next_day = current_time + timedelta(days=1)
    next_day = next_day.replace(hour=6, minute=30, second=0, microsecond=0)
    
    while next_day.weekday() >= 5:  # Saturday or Sunday
        next_day += timedelta(days=1)
    
    return next_day

@bot.command()
async def track(ctx, symbol: str):
    symbol = symbol.upper()
    user_id = ctx.author.id

    if user_id not in user_tracking:
        user_tracking[user_id] = set()
    user_tracking[user_id].add(symbol)

    if symbol not in symbol_subscribers:
        symbol_subscribers[symbol] = set()
    symbol_subscribers[symbol].add(user_id)

    async with aiohttp.ClientSession() as session:
        quote_data = await get_fmp_data(session, 'quote', symbol)
        if quote_data:
            current_price = quote_data[0]['price']
            change = quote_data[0]['change']
            change_percent = quote_data[0]['changesPercentage']

            now_pst = datetime.now(pytz.timezone('US/Pacific'))
            date_str = now_pst.strftime('%Y-%m-%d')
            time_str = now_pst.strftime('%I:%M %p PST')

            message = (
                f"{ctx.author.mention}, now tracking **{symbol}** for you. Here's the current information:\n\n"
                f"**Detailed information for {symbol}:**\n"
                f"Date: {date_str}\n"
                f"Time: {time_str}\n"
                f"Current Price: ${current_price:.2f}\n"
                f"Change: ${change:.2f} ({change_percent:.2f}%)"
            )
        else:
            message = f"{ctx.author.mention}, you are now tracking **{symbol}**, but I couldn't fetch current data."

    await ctx.send(message)

@bot.command()
async def untrack(ctx, symbol: str):
    symbol = symbol.upper()
    user_id = ctx.author.id

    if user_id in user_tracking and symbol in user_tracking[user_id]:
        user_tracking[user_id].remove(symbol)
        if not user_tracking[user_id]:
            del user_tracking[user_id]

        symbol_subscribers[symbol].remove(user_id)
        if not symbol_subscribers[symbol]:
            del symbol_subscribers[symbol]
            last_alert_times.pop(symbol, None)
            last_alert_prices.pop(symbol, None)

        message = f"{ctx.author.mention}, you have stopped tracking **{symbol}**."
    else:
        message = f"{ctx.author.mention}, you are not tracking **{symbol}**."

    await ctx.send(message)

@bot.command(name='list')
async def list_tracked(ctx):
    user_id = ctx.author.id
    symbols = user_tracking.get(user_id, set())
    if symbols:
        symbols_list = ', '.join(f'**{s}**' for s in symbols)
        message = f"{ctx.author.mention}, you are currently tracking: {symbols_list}"
    else:
        message = f"{ctx.author.mention}, you are not tracking any symbols."

    await ctx.send(message)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print(f"Bot is ready to monitor stocks")
    bot.loop.create_task(monitor_stocks())

def run_discord_bot():
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    run_discord_bot()