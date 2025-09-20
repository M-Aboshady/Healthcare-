import streamlit as st
import pandas as pd

st.title("🧑‍⚕️ Weekly Patient Journey Suggestions (Lab ↔ Pharmacy)")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
LAB_THRESHOLD = st.sidebar.number_input(
    "Lab crowding threshold (average tokens)", min_value=1, max_value=200, value=10, step=1
)
PHARM_THRESHOLD = st.sidebar.number_input(
    "Pharmacy crowding threshold (average tokens)", min_value=1, max_value=200, value=20, step=1
)

# 🔥 Flexible aggregation
TIME_WINDOW = st.sidebar.number_input(
    "Aggregation window (minutes)",
    min_value=15,
    max_value=1440,
    value=30,
    step=15,
    help="Examples: 30 = half hour, 60 = 1 hour, 180 = 3 hours, 1440 = full day",
)

# Weekday filter
WEEKDAYS = ["All Days", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_day = st.sidebar.selectbox("Filter by weekday", WEEKDAYS)

# Start hour filter
selected_hour = st.sidebar.slider(
    "Start hour of the day (0–23)", min_value=0, max_value=23, value=0, step=1
)

# Upload files
lab_file = st.file_uploader("Upload Lab Data CSV", type=["csv"])
pharm_file = st.file_uploader("Upload Pharmacy Data CSV", type=["csv"])

if lab_file and pharm_file:
    # Load CSVs
    lab = pd.read_csv(lab_file)
    pharm = pd.read_csv(pharm_file)

    # Parse datetime columns
    lab["Date"] = pd.to_datetime(lab["Date"])
    pharm["Date"] = pd.to_datetime(pharm["Date"])

    # Create day + slot
    def create_slot(df, date_col, time_col, window):
        df["day"] = df[date_col].dt.day_name()
        df["hour"] = pd.to_datetime(df[time_col]).dt.hour
        df["slot"] = (df["hour"] * 60 + pd.to_datetime(df[time_col]).dt.minute) // window
        return df

    lab = create_slot(lab, "Date", "time", TIME_WINDOW)
    pharm = create_slot(pharm, "Date", "time", TIME_WINDOW)

    # Apply weekday filter
    if selected_day != "All Days":
        lab = lab[lab["day"] == selected_day]
        pharm = pharm[pharm["day"] == selected_day]

    # Apply start hour filter
    lab = lab[lab["hour"] >= selected_hour]
    pharm = pharm[pharm["hour"] >= selected_hour]

    # Aggregate average Token No per day + slot
    lab_agg = lab.groupby(["day", "slot"])["Token No"].mean().reset_index(name="lab_tokens")
    pharm_agg = pharm.groupby(["day", "slot"])["Token No"].mean().reset_index(name="pharm_tokens")

    # Merge datasets
    merged = pd.merge(lab_agg, pharm_agg, on=["day", "slot"], how="outer").fillna(0)

    st.subheader("📅 Weekly Suggestions (based on patient counts)")
    suggestions = []

    for _, row in merged.iterrows():
        day, slot = row["day"], row["slot"]
        lab_tokens, pharm_tokens = row["lab_tokens"], row["pharm_tokens"]

        # Convert slot back to readable time
        start_minutes = slot * TIME_WINDOW
        end_minutes = (slot + 1) * TIME_WINDOW
        start_hour, start_min = divmod(start_minutes, 60)
        end_hour, end_min = divmod(end_minutes, 60)

        if TIME_WINDOW == 1440:
            slot_label = f"{day} (full day)"
        else:
            slot_label = f"{day} {start_hour:02d}:{start_min:02d}–{end_hour:02d}:{end_min:02d}"

        # Decision rules
        if pharm_tokens > PHARM_THRESHOLD and lab_tokens <= LAB_THRESHOLD:
            suggestions.append(
                f"{slot_label} → Pharmacy crowded ({pharm_tokens:.1f} tokens). Lab free ({lab_tokens:.1f} tokens). Suggest: shift patients to Lab."
            )
        elif lab_tokens > LAB_THRESHOLD and pharm_tokens <= PHARM_THRESHOLD:
            suggestions.append(
                f"{slot_label} → Lab crowded ({lab_tokens:.1f} tokens). Pharmacy free ({pharm_tokens:.1f} tokens). Suggest: shift patients to Pharmacy."
            )
        elif pharm_tokens > PHARM_THRESHOLD and lab_tokens > LAB_THRESHOLD:
            suggestions.append(
                f"{slot_label} → Both Lab ({lab_tokens:.1f} tokens) and Pharmacy ({pharm_tokens:.1f} tokens) crowded. Suggest: add staff or reschedule."
            )
        else:
            continue

    if suggestions:
        for s in suggestions:
            st.write("✅ " + s)
    else:
        st.info("No critical crowding detected for this selection. ✅")
