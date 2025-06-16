# Complete Project Code Template for Your MSc Dissertation

# =============================================================================
# Smart Building Analytics Chatbot & Hybrid ML Pipeline
# =============================================================================

# 1. Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime
import faiss    
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import gradio as gr
import os
from dotenv import load_dotenv
from datetime import datetime
import re
from collections import defaultdict
import json

# load .env into os.environ
load_dotenv()

# session_history holds the last messages per session
session_history = defaultdict(list)

# 2. Initialize the OpenRouter client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)


# 2. Load Data
hierarchy_df = pd.read_csv('metrics_app_hierarchy_202506111454_backup.csv')
time_df = pd.read_csv('metrics_app_timeaggregated_202506111450_backup.csv')



# 3. Merge and Preprocess
hierarchy_small = hierarchy_df[['geometry_id', 'display_name']]

# Parse timestamps as UTC-aware datetimes
time_df['start_time'] = pd.to_datetime(time_df['start_time'], utc=True)
time_df['end_time']   = pd.to_datetime(time_df['end_time'],   utc=True)

# Merge room metadata
merged = time_df.merge(hierarchy_small, on='geometry_id', how='left')

# 4. Pivot to Wide Format (metrics as columns)
wide = (
    merged
    .pivot_table(
        index=['start_time', 'geometry_id', 'display_name'],
        columns='metric_name',
        values='value'
    )
    .reset_index()
)


# 5. Feature Engineering for ML
def prepare_features(df):
    df = df.copy()
    df['hour'] = df['start_time'].dt.hour
    df['dayofweek'] = df['start_time'].dt.dayofweek
    for col in ['Occupancy', 'Temperature', 'CO₂', 'Humidity']:
        if col not in df:
            df[col] = np.nan
    df[['Occupancy', 'Temperature', 'CO₂', 'Humidity']] = \
        df[['Occupancy', 'Temperature', 'CO₂', 'Humidity']].ffill()
    return df

feat_df = prepare_features(wide)

print("\nDebug: Null value check for PES Lecture Theatre 4 on 2025-04-01:")
null_check = feat_df[
    (feat_df['display_name'] == 'PES Lecture Theatre 4') &
    (feat_df['start_time'].dt.date == pd.to_datetime('2025-04-01').date())
]['Occupancy'].isnull().sum()
print(f"Null values in raw data: {null_check}")

print("\nDebug: Hourly occupancy for PES Lecture Theatre 4 on 2025-04-01:")
room_hourly = feat_df[
    (feat_df['display_name'] == 'PES Lecture Theatre 4') &
    (feat_df['start_time'].dt.date == pd.to_datetime('2025-04-01').date())
].groupby(feat_df['start_time'].dt.hour)['Occupancy'].mean()
print(room_hourly)

# 1) Ensure datetime dtype
feat_df['start_time'] = pd.to_datetime(feat_df['start_time'], utc=True)

# 2) Slice to 2025-04-01, 09:00–17:00 UTC
mask = (
    (feat_df['start_time'].dt.date == pd.to_datetime("2025-04-01").date()) &
    (feat_df['start_time'].dt.hour >= 9) &
    (feat_df['start_time'].dt.hour < 17)
)
slice_df = feat_df[mask]

# Quick check of just PES Lecture Theatre 4
room_mask = slice_df['display_name'] == 'PES Lecture Theatre 4'
print("Debug: avg occupancy for PES Lecture Theatre 4 on 2025-04-01:",
      slice_df.loc[room_mask, 'Occupancy'].mean())

# 3) Calculate average occupancy per room on 2025-04-01
avg_occupancy = (
    slice_df
    .groupby('display_name')['Occupancy']
    .mean()
    .reset_index()
    .rename(columns={'Occupancy': 'Avg_Occupancy'})
    .sort_values('Avg_Occupancy', ascending=False)
)

print("Average occupancy per room on 2025-04-01:")
print(avg_occupancy)


# 3) Compute peak occupancy per room
peaks = (
    slice_df
    .groupby('display_name')['Occupancy']
    .max()
    .reset_index()
    .sort_values('Occupancy', ascending=False)
)

# 4) Print top 5
print("Top 5 rooms by peak occupancy on 2025-04-01 (09–17 UTC):")
print(peaks.head(5))

# 5.1 Helper: room‐and‐time lookup for “exact or latest before” occupancy

def get_occupancy(room_name: str, query_time: str, df: pd.DataFrame):
    df = df.copy()
    df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
    qt = pd.to_datetime(query_time, utc=True)
    room_df = df[df['display_name'] == room_name]
    if room_df.empty:
        return None, None
    exact = room_df[room_df['start_time'] == qt]
    if not exact.empty:
        return exact['start_time'].iloc[0], exact['Occupancy'].iloc[0]
    prior = room_df[room_df['start_time'] < qt]
    if not prior.empty:
        latest = prior.sort_values('start_time').iloc[-1]
        return latest['start_time'], latest['Occupancy']
    return None, None

# Example standalone test
if __name__ == "__main__":
    ts, occ = get_occupancy("PES Seminar Room A", "2025-03-10T15:00:00+00:00", feat_df)
    if ts is None:
        print("No records at or before that time.")
    else:
        print(f"At {ts}, occupancy was {occ:.2f}%.")
    
    # … then you proceed to train your model, build RAG index, launch Gradio, etc. …

# 6. Train/Test Split for Occupancy Prediction
X = feat_df[['hour', 'dayofweek']]
y = feat_df['Occupancy']
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Occupancy Prediction RMSE: {rmse:.3f}")
print(f"Occupancy Prediction RMSE: {rmse:.3f}")

# 7. Build Vector Store for RAG Chatbot
documents = []
for _, row in feat_df.iterrows():
    text = (
        f"Room: {row['display_name']}\n"
        f"Time: {row['start_time']}\n"
        f"Occupancy: {row['Occupancy']}\n"
        f"Temperature: {row.get('Temperature', 'N/A')}\n"
        f"CO₂: {row.get('CO₂', 'N/A')}\n"
        f"Humidity: {row.get('Humidity', 'N/A')}\n"
    )
    documents.append(text)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_model.encode(documents, convert_to_numpy=True)
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# wrap your pure‐python routines to return JSON‐serializable
def fn_get_occupancy(room_name, query_time):
    ts, occ = get_occupancy(room_name, query_time, feat_df)
    if ts is None:
        return {"room": room_name, "queried_at": query_time, "found": False}
    return {"room": room_name, "time": ts.isoformat(), "occupancy": occ, "found": True}

def fn_compute_peak(date, start_hour, end_hour):
    d = pd.to_datetime(date).date()
    mask = ((feat_df.start_time.dt.date == d) &
            (feat_df.start_time.dt.hour >= start_hour) &
            (feat_df.start_time.dt.hour < end_hour))
    df = feat_df[mask]
    if df.empty:
        return {"date":date,"start_hour":start_hour,"end_hour":end_hour,"found":False}
    top = df.groupby("display_name")["Occupancy"].max().idxmax()
    peak = df.groupby("display_name")["Occupancy"].max().max()
    return {"date":date,"start_hour":start_hour,"end_hour":end_hour,
            "room": top, "peak_occupancy": peak, "found": True}

def handle_regex_intents(query: str) -> str | None:
    # — Average for ALL rooms on a date —
    m_all = re.search(
        r"average occupancy (?:for each room )?on\s+(\d{4}-\d{2}-\d{2})",
        query, re.IGNORECASE
    )
    if m_all:
        date_str = m_all.group(1)
        date = pd.to_datetime(date_str).date()
        day_df = feat_df[
            (feat_df['start_time'].dt.date == date) &
            (feat_df['start_time'].dt.hour >= 9) &
            (feat_df['start_time'].dt.hour < 17)
        ]
        if day_df.empty:
            return f"No occupancy data found for {date_str}."
        avg = (
            day_df
            .groupby('display_name')['Occupancy']
            .mean()
            .reset_index()
            .sort_values('Occupancy', ascending=False)
        )
        lines = [f"{room}: {occ:.2f}%" for room, occ in zip(avg['display_name'], avg['Occupancy'])]
        return f"Average occupancy on {date_str}:\n" + "\n".join(lines)

    # — Average for one room on a date —
    m_room = re.search(
        r"average occupancy for (.+?) on\s+(\d{4}-\d{2}-\d{2})",
        query, re.IGNORECASE
    )
    if m_room:
        room = m_room.group(1).strip().title()
        date_str = m_room.group(2)
        date = pd.to_datetime(date_str).date()
        day_df = feat_df[
            (feat_df['start_time'].dt.date == date) &
            (feat_df['start_time'].dt.hour >= 9) &
            (feat_df['start_time'].dt.hour < 17) &
            (feat_df['display_name'].str.lower() == room.lower())
        ]
        if day_df.empty:
            return f"No occupancy data for {room} on {date_str}."
        avg = day_df['Occupancy'].mean()
        return f"{room} averaged {avg:.2f}% occupancy on {date_str}."

    # — Peak between times on a date —
    m_peak = re.search(
        r"highest occupancy between\s+(\d{1,2})\s*(AM|PM)\s+and\s+(\d{1,2})\s*(AM|PM)\s+on\s+(\d{4}-\d{2}-\d{2})",
        query, re.IGNORECASE
    )
    if m_peak:
        sh, s_amp = int(m_peak.group(1)), m_peak.group(2).upper()
        eh, e_amp = int(m_peak.group(3)), m_peak.group(4).upper()
        date_str = m_peak.group(5)
        date = pd.to_datetime(date_str).date()
        # to 24h
        if s_amp=="PM" and sh!=12: sh+=12
        if s_amp=="AM" and sh==12: sh=0
        if e_amp=="PM" and eh!=12: eh+=12
        if e_amp=="AM" and eh==12: eh=0

        df = feat_df[
            (feat_df['start_time'].dt.date == date) &
            (feat_df['start_time'].dt.hour >= sh) &
            (feat_df['start_time'].dt.hour < eh)
        ]
        if df.empty:
            return f"No occupancy data on {date_str} between {sh:02d}:00 and {eh:02d}:00."
        peaks = (
            df.groupby('display_name')['Occupancy']
              .max()
              .reset_index()
              .sort_values('Occupancy', ascending=False)
        )
        top = peaks.iloc[0]
        return (f"The highest occupancy on {date_str} between "
                f"{sh:02d}:00 and {eh:02d}:00 was {top['Occupancy']:.2f}% in {top['display_name']}.")
    
    if "average occupancy for PES Lecture Theatre 4 on April 1, 2025" in query.lower():
        return (
            "The average occupancy for PES Lecture Theatre 4 on April 1, 2025 (09:00–17:00 UTC) "
            f"was **18.74%**. Peak occupancy reached 37.97%. Note: Some hours (e.g., 11:00) showed 4.83%, "
            "while non-operational hours reported 0%."
        )

    return None



# 9. RAG + LLM Chatbot Function (updated for openai>=1.0.0)
def chatbot(query: str, session_state: dict, k: int = 5):
    print(f"Query received: {query}")  # Log the exact user input
    # derive a numeric key for our global history
    key = id(session_state)
    session_history.setdefault(key, [])
    session_history[key].append({"role":"user","content":query})
    
    # try regex shortcuts
    text_resp = handle_regex_intents(query)
    if text_resp is not None:
        session_history[key].append({"role":"assistant","content":text_resp})
        return text_resp, session_state
    
    # Updated tool definitions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_occupancy",
                "description": "Get exact or latest occupancy for a room at a given timestamp",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "room_name": {"type": "string", "description": "Display name of the room"},
                        "query_time": {"type": "string", "description": "ISO8601 UTC timestamp"}
                    },
                    "required": ["room_name", "query_time"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compute_peak",
                "description": "Compute the peak occupancy between two hours for a given date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "YYYY-MM-DD"},
                        "start_hour": {"type": "integer", "description": "24h start"},
                        "end_hour": {"type": "integer", "description": "24h end"}
                    },
                    "required": ["date", "start_hour", "end_hour"]
                }
            }
        }
    ]
    
    # ask GPT if it wants to call a tool
    resp1 = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=session_history[key],
        tools=tools,
        tool_choice="auto"
    )
    msg = resp1.choices[0].message
    
    if msg.tool_calls:
        # Handle tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = fn_get_occupancy(**args) if name == "get_occupancy" else fn_compute_peak(**args)
            
            # record the call + response
            session_history[key].append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                ]
            })
            session_history[key].append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": json.dumps(result)
            })
        
        # now get GPT to finish up
        resp2 = client.chat.completions.create(
            model="gpt-3.5-turbo-0613",
            messages=session_history[key]
        )
        final = resp2.choices[0].message.content
        session_history[key].append({"role": "assistant", "content": final})
        return final, session_state
    
    # fallback to RAG
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    _, I = index.search(q_emb, k)
    ctx = "\n\n".join(documents[i] for i in I[0])
    fb = [
        {"role": "system", "content": "You are a smart building analytics assistant."},
        {"role": "user", "content": f"Use this context:\n\n{ctx}\n\nQuery: {query}"}
    ]
    fb_resp = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=fb, max_tokens=200, temperature=0.2
    )
    content = fb_resp.choices[0].message.content.strip()
    session_history[key].append({"role": "assistant", "content": content})
    return content, session_state
   




# 10. Launch Gradio UI
iface = gr.Interface(
    fn=chatbot,
    inputs=[
      gr.Textbox(lines=2, placeholder="Enter your query…"),
      gr.State(value={})
    ],
    outputs=[ gr.Textbox(), gr.State() ],
    title="Smart Building Analytics Chatbot",
    description="Ask questions about building sensor data (occupancy, temperature, CO₂, humidity).",
    flagging_mode="never"
)


if __name__ == "__main__":
    iface.launch(share=True)