import streamlit as st
import pandas as pd

st.title("🧑‍⚕️ Weekly Patient Journey Suggestions (Lab ↔ Pharmacy)")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
LAB_THRESHOLD = st.sidebar.number_input(
    "Lab crowding threshold (minutes)", min_value=1, max_value=60, value=5, step=1
)
PHARM_THRESHOLD = st.sidebar.number_input(
    "Pharmacy crowding threshold (minutes)", min_value=1, max_value=60, value=10, step=1
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

# Convert waiting_time to minutes if it looks like a time string
def convert_waiting_time(df):
    if df["waiting_time"].dtype == object:  # not numeric
        try:
            # Try parsing as timedelta
            df["waiting_time"] = pd.to_timedelta(df["waiting_time"]).dt.total_seconds() / 60
        except:
            # If already numeric-like but string, force convert
            df["waiting_time"] = pd.to_numeric(df["waiting_time"], errors="coerce")
    return df

lab = convert_waiting_time(lab)
pharm = convert_waiting_time(pharm)


if lab_file and pharm_file:
    # Load CSVs
    lab = pd.read_csv(lab_file)
    pharm = pd.read_csv(pharm_file)
    

    # Ensure datetime column exists (adjust column name if needed)
    lab["time"] = pd.to_datetime(lab["time"])
    pharm["time"] = pd.to_datetime(pharm["time"])

    # Create day + time slots
    def create_slot(df, time_col, window):
        df["day"] = df[time_col].dt.day_name()
        df["slot"] = (df[time_col].dt.hour * 60 + df[time_col].dt.minute) // window
        df["hour"] = df[time_col].dt.hour
        return df

    lab = create_slot(lab, "time", TIME_WINDOW)
    pharm = create_slot(pharm, "time", TIME_WINDOW)

    # Apply weekday filter (if not "All Days")
    if selected_day != "All Days":
        lab = lab[lab["day"] == selected_day]
        pharm = pharm[pharm["day"] == selected_day]

    # Apply start hour filter
    lab = lab[lab["hour"] >= selected_hour]
    pharm = pharm[pharm["hour"] >= selected_hour]

    # Aggregate avg waiting times per day + slot
    lab_agg = lab.groupby(["day", "slot"])["waiting_time"].mean().reset_index(name="lab_wait")
    pharm_agg = pharm.groupby(["day", "slot"])["waiting_time"].mean().reset_index(name="pharm_wait")

    # Merge datasets
    merged = pd.merge(lab_agg, pharm_agg, on=["day", "slot"], how="outer").fillna(0)

    st.subheader("📅 Weekly Suggestions")
    suggestions = []

    for _, row in merged.iterrows():
        day, slot = row["day"], row["slot"]
        lab_wait, pharm_wait = row["lab_wait"], row["pharm_wait"]

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
        if pharm_wait > PHARM_THRESHOLD and lab_wait <= LAB_THRESHOLD:
            suggestions.append(
                f"{slot_label} → Pharmacy crowded ({pharm_wait:.1f} min). Lab free ({lab_wait:.1f} min). Suggest: shift patients to Lab."
            )
        elif lab_wait > LAB_THRESHOLD and pharm_wait <= PHARM_THRESHOLD:
            suggestions.append(
                f"{slot_label} → Lab crowded ({lab_wait:.1f} min). Pharmacy free ({pharm_wait:.1f} min). Suggest: shift patients to Pharmacy."
            )
        elif pharm_wait > PHARM_THRESHOLD and lab_wait > LAB_THRESHOLD:
            suggestions.append(
                f"{slot_label} → Both Lab ({lab_wait:.1f} min) and Pharmacy ({pharm_wait:.1f} min) crowded. Suggest: add staff or reschedule."
            )
        else:
            continue

    if suggestions:
        for s in suggestions:
            st.write("✅ " + s)
    else:
        st.info("No critical crowding detected for this selection. ✅")
