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
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import re

# load .env into os.environ
load_dotenv()

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

# 1) Ensure datetime dtype
feat_df['start_time'] = pd.to_datetime(feat_df['start_time'], utc=True)

# 2) Slice to 2025-04-01, 09:00–17:00 UTC
mask = (
    (feat_df['start_time'].dt.date == pd.to_datetime("2025-04-01").date()) &
    (feat_df['start_time'].dt.hour >= 9) &
    (feat_df['start_time'].dt.hour < 17)
)
slice_df = feat_df[mask]

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



# 9. RAG + LLM Chatbot Function (updated for openai>=1.0.0)
def chatbot(query: str, k: int = 5) -> str:
    # A) Average‐occupancy for all rooms on a date
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

    # B) Average‐occupancy for a specific room on a date
    m_room = re.search(
        r"average occupancy for (.+?) on\s+("
            r"\d{4}-\d{2}-\d{2}"           # 2025-04-01
            r"|[A-Za-z]+ \d{1,2}, \d{4}"  # April 1, 2025
        r")",
        query, re.IGNORECASE
    )
    if m_room:
        # 1) clean up the room name
        room     = m_room.group(1).strip()
        room     = re.sub(r'^(the)\s+', '', room, flags=re.IGNORECASE)
        room     = room.title()
        date_str = m_room.group(2)
        date     = pd.to_datetime(date_str).date()

        # 2) filter
        mask = (
           (feat_df['start_time'].dt.date  == date) &
           (feat_df['start_time'].dt.hour  >=  9) &
           (feat_df['start_time'].dt.hour  <  17) &
           (feat_df['display_name'].str.lower() == room.lower())
       )

        day_df   = feat_df[mask]
        if day_df.empty:
            return f"No occupancy data for {room} on {date_str}."
        avg      = day_df['Occupancy'].mean()
        return f"{room} averaged {avg:.2f}% occupancy on {date_str}."


    # — C) Peak‐occupancy intent —
    m_peak = re.search(
        r"highest occupancy between\s+(\d{1,2})\s*(AM|PM)\s+and\s+(\d{1,2})\s*(AM|PM)\s+on\s+(\d{4}-\d{2}-\d{2})",
        query, re.IGNORECASE
    )
    if m_peak:
        start_h, start_ampm = int(m_peak.group(1)), m_peak.group(2).upper()
        end_h,   end_ampm   = int(m_peak.group(3)), m_peak.group(4).upper()
        date_str = m_peak.group(5)
        date = pd.to_datetime(date_str).date()

        # convert to 24h
        if start_ampm=="PM" and start_h!=12: start_h+=12
        if start_ampm=="AM" and start_h==12: start_h=0
        if end_ampm  =="PM" and end_h  !=12: end_h  +=12
        if end_ampm  =="AM" and end_h  ==12: end_h  =0

        mask = (
            (feat_df['start_time'].dt.date == date) &
            (feat_df['start_time'].dt.hour >= start_h) &
            (feat_df['start_time'].dt.hour <  end_h)
        )
        slice_df = feat_df[mask]
        if slice_df.empty:
            return f"No occupancy data on {date_str} between {start_h:02d}:00 and {end_h:02d}:00."

        peaks = (
            slice_df
            .groupby('display_name')['Occupancy']
            .max()
            .reset_index()
            .sort_values('Occupancy', ascending=False)
        )
        top = peaks.iloc[0]
        return (f"The highest occupancy on {date_str} between "
                f"{start_h:02d}:00 and {end_h:02d}:00 was {top['Occupancy']:.2f}% "
                f"in {top['display_name']}.")

    # — D) Fallback to RAG + LLM —
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    _, I = index.search(q_emb, k)
    context = "\n\n".join(documents[i] for i in I[0])
    messages = [
        {"role": "system", "content": "You are a smart building analytics assistant."},
        {"role": "user",   "content": f"Use this context:\n\n{context}\n\nQuery: {query}"}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()




# 10. Launch Gradio UI
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Enter your technical query..."),
    outputs="text",
    title="Smart Building Analytics Chatbot",
    description="Ask questions about building sensor data (occupancy, temperature, CO₂, humidity).",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch(share=True)