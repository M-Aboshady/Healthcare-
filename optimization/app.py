import streamlit as st
import pandas as pd

st.title("🧑‍⚕️ Weekly Patient Journey Suggestions (Lab ↔ Pharmacy)")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
LAB_THRESHOLD = st.sidebar.number_input(
    "Lab crowding threshold (tokens)", min_value=1, max_value=100, value=10, step=1
)
PHARM_THRESHOLD = st.sidebar.number_input(
    "Pharmacy crowding threshold (tokens)", min_value=1, max_value=100, value=20, step=1
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

# NEW: Weekday filter
WEEKDAYS = ["All Days", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_day = st.sidebar.selectbox("Filter by weekday", WEEKDAYS)

# NEW: Start hour filter
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

    # Correct: Use the 'Date' column to create the day name
    lab["Date"] = pd.to_datetime(lab["Date"])
    pharm["Date"] = pd.to_datetime(pharm["Date"])

    # Create day + time slots
    def create_slot(df, date_col, time_col, window):
        df["day"] = df[date_col].dt.day_name()
        df["hour"] = pd.to_datetime(df[time_col]).dt.hour
        df["slot"] = (df["hour"] * 60 + pd.to_datetime(df[time_col]).dt.minute) // window
        return df

    lab = create_slot(lab, "Date", "time", TIME_WINDOW)
    pharm = create_slot(pharm, "Date", "time", TIME_WINDOW)

    # Convert waiting_time
    def convert_waiting_time(df):
        if df["waiting_time"].dtype == object:
            try:
                df["waiting_time"] = pd.to_timedelta(df["waiting_time"]).dt.total_seconds() / 60
            except:
                df["waiting_time"] = pd.to_numeric(df["waiting_time"], errors="coerce")
        return df
    
    lab = convert_waiting_time(lab)
    pharm = convert_waiting_time(pharm)

    # Apply weekday filter (if not "All Days")
    if selected_day != "All Days":
        lab = lab[lab["day"] == selected_day]
        pharm = pharm[pharm["day"] == selected_day]

    # Apply start hour filter
    lab = lab[lab["hour"] >= selected_hour]
    pharm = pharm[pharm["hour"] >= selected_hour]

    # Aggregate avg waiting times per day + slot
    lab_agg = lab.groupby(["day", "slot"])["Token No"].count().reset_index(name="lab_count")
    pharm_agg = pharm.groupby(["day", "slot"])["Token No"].count().reset_index(name="pharm_count")

    # Merge datasets
    merged = pd.merge(lab_agg, pharm_agg, on=["day", "slot"], how="outer").fillna(0)

    st.subheader("📅 Weekly Suggestions")
    suggestions = []

    for _, row in merged.iterrows():
        day, slot = row["day"], row["slot"]
        lab_count, pharm_count = row["lab_count"], row["pharm_count"]

        # Convert slot back to readable time
        start_minutes = slot * TIME_WINDOW
        end_minutes = (slot + 1) * TIME_WINDOW
        start_hour, start_min = divmod(start_minutes, 60)
        end_hour, end_min = divmod(end_minutes, 60)

        if TIME_WINDOW == 1440:  # Special case: whole day
            slot_label = f"{day} (full day)"
        else:
            slot_label = f"{day} {start_hour:02d}:{start_min:02d}–{end_hour:02d}:{end_min:02d}"

        # Decision rules
        if pharm_count > PHARM_THRESHOLD and lab_count <= LAB_THRESHOLD:
            suggestions.append(
                f"{slot_label} → Pharmacy crowded ({pharm_count:.0f} tokens). Lab free ({lab_count:.0f} tokens). Suggest: shift patients to Lab."
            )
        elif lab_count > LAB_THRESHOLD and pharm_count <= PHARM_THRESHOLD:
            suggestions.append(
                f"{slot_label} → Lab crowded ({lab_count:.0f} tokens). Pharmacy free ({pharm_count:.0f} tokens). Suggest: shift patients to Pharmacy."
            )
        elif pharm_count > PHARM_THRESHOLD and lab_count > LAB_THRESHOLD:
            suggestions.append(
                f"{slot_label} → Both Lab ({lab_count:.0f} tokens) and Pharmacy ({pharm_count:.0f} tokens) crowded. Suggest: add staff or reschedule."
            )
        else:
            continue

    if suggestions:
        for s in suggestions:
            st.write("✅ " + s)
    else:
        st.info("No critical crowding detected for this selection. ✅")
