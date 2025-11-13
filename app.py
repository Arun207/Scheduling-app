import io
import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


DEFAULT_SHIFT_CONFIG = [
    {"Shift": "Morning", "Headcount": 2},
    {"Shift": "Evening", "Headcount": 2},
    {"Shift": "Night", "Headcount": 1},
]
DEFAULT_WEEK_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    """Load the baked-in sample data as a starting point."""
    sources = [
        ("Schedule_aw.xlsx", pd.read_excel, {"engine": None}),
        ("employee_total_hours.csv", pd.read_csv, {}),
    ]

    for path, loader, kwargs in sources:
        try:
            df = loader(path, **{k: v for k, v in kwargs.items() if v is not None})
            st.success(f"Loaded sample data from `{path}`.")
            return df
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read `{path}` ({exc}). Trying next fallback…")

    st.info("No bundled data found. Starting with an empty template instead.")
    return pd.DataFrame(
        {
            "Name": [],
            "Total Hours": [],
            "Available Days": [],
            "Shift Preference": [],
            "Fulltime/Part-time": [],
            "Tier": [],
        }
    )


def read_uploaded_file(upload) -> pd.DataFrame:
    """Handle CSV/Excel uploads provided by the user."""
    if upload is None:
        return load_default_data()

    buffer = io.BytesIO(upload.getbuffer())
    name = upload.name.lower()

    try:
        if name.endswith(".csv"):
            return pd.read_csv(buffer)
        if name.endswith((".xls", ".xlsx")):
            return pd.read_excel(buffer)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to parse `{upload.name}` — {exc}")
        return load_default_data()

    st.error("Unsupported file type. Please upload a CSV or Excel file.")
    return load_default_data()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring multiple naming styles down to a single canonical schema."""
    rename_map = {
        "employee": "Name",
        "employee_name": "Name",
        "name": "Name",
        "total_hours": "Total Hours",
        "hours": "Total Hours",
        "available_days": "Available Days",
        "availability": "Available Days",
        "shift_preference": "Shift Preference",
        "preferred_shift": "Shift Preference",
        "employment_type": "Fulltime/Part-time",
        "status": "Fulltime/Part-time",
        "tier_level": "Tier",
    }
    cleaned = df.copy()
    cleaned.columns = [rename_map.get(str(col).strip().lower(), col) for col in cleaned.columns]

    required = [
        "Name",
        "Total Hours",
        "Available Days",
        "Shift Preference",
        "Fulltime/Part-time",
        "Tier",
    ]
    for col in required:
        if col not in cleaned.columns:
            if col == "Total Hours":
                cleaned[col] = 0
            elif col == "Tier":
                cleaned[col] = "Tier 3"
            elif col == "Shift Preference":
                cleaned[col] = "Morning"
            elif col == "Available Days":
                cleaned[col] = "Mon,Tue,Wed,Thu,Fri"
            elif col == "Fulltime/Part-time":
                cleaned[col] = "Fulltime"
            else:
                cleaned[col] = ""
    return cleaned[required]


def parse_available_days(value) -> List[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(v).strip().title()[:3] for v in value if str(v).strip()]
    value = str(value)
    separators = [",", ";", "|", "/"]
    for sep in separators:
        if sep in value:
            parts = [p.strip() for p in value.split(sep)]
            break
    else:
        parts = [value.strip()]
    return [p.title()[:3] for p in parts if p]


def tier_to_rank(tier_value) -> int:
    if pd.isna(tier_value):
        return 999
    match = re.search(r"(\d+)", str(tier_value))
    if match:
        return int(match.group(1))
    lookup = {"tier a": 1, "tier b": 2, "tier c": 3}
    return lookup.get(str(tier_value).strip().lower(), 999)


def build_employee_state(df: pd.DataFrame, shift_hours: int) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["available_days_list"] = cleaned["Available Days"].apply(parse_available_days)
    cleaned["shift_pref_norm"] = cleaned["Shift Preference"].fillna("").str.strip().str.title()
    cleaned["tier_rank"] = cleaned["Tier"].apply(tier_to_rank)
    cleaned["remaining_hours"] = (
        pd.to_numeric(cleaned["Total Hours"], errors="coerce").fillna(0).clip(lower=0)
    )
    cleaned["max_assignments"] = (cleaned["remaining_hours"] // shift_hours).astype(int)
    cleaned["scheduled_hours"] = 0
    cleaned["assignments"] = [[] for _ in range(len(cleaned))]
    return cleaned


def normalise_shift_config(entries: List[Dict]) -> List[Dict]:
    cleaned: List[Dict] = []
    for entry in entries:
        raw_shift = entry.get("Shift", "")
        if pd.isna(raw_shift):
            raw_shift = ""
        shift_name = str(raw_shift).strip()
        if not shift_name or shift_name.lower() == "nan":
            continue

        raw_headcount = entry.get("Headcount", 0)
        if pd.isna(raw_headcount) or raw_headcount == "":
            headcount = 0
        else:
            try:
                headcount = int(float(raw_headcount))
            except (TypeError, ValueError):
                headcount = 0

        cleaned.append({"Shift": shift_name.title(), "Headcount": max(0, headcount)})
    return cleaned


def _eligible_candidates(
    employees: pd.DataFrame,
    day: str,
    shift: str,
    require_preference: bool,
    shift_hours: int,
    assigned_today: Dict[str, set],
) -> pd.DataFrame:
    mask = employees["available_days_list"].apply(lambda ds: day in ds if ds else True)
    mask &= employees["remaining_hours"] >= shift_hours
    mask &= employees["max_assignments"] > employees["assignments"].apply(len)
    mask &= ~employees["Name"].isin(assigned_today.get(day, set()))

    if require_preference:
        mask &= employees["shift_pref_norm"] == shift

    eligible = employees[mask].copy()
    if eligible.empty:
        return eligible

    eligible = eligible.sort_values(
        by=["tier_rank", "remaining_hours", "Fulltime/Part-time"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return eligible


def assign_shift(
    employees: pd.DataFrame,
    day: str,
    shift: str,
    headcount: int,
    shift_hours: int,
    assigned_today: Dict[str, set],
) -> Tuple[List[Dict], pd.DataFrame]:
    remaining_slots = headcount
    assignments: List[Dict] = []

    for preference_pass in (True, False):
        if remaining_slots <= 0:
            break

        candidates = _eligible_candidates(
            employees, day, shift, preference_pass, shift_hours, assigned_today
        )

        for _, row in candidates.iterrows():
            if remaining_slots <= 0:
                break
            idx = row.name
            employees.at[idx, "remaining_hours"] -= shift_hours
            employees.at[idx, "scheduled_hours"] += shift_hours
            employees.at[idx, "assignments"].append({"day": day, "shift": shift})
            assigned_today.setdefault(day, set()).add(row["Name"])

            assignments.append(
                {
                    "Day": day,
                    "Shift": shift,
                    "Employee": row["Name"],
                    "Tier": row["Tier"],
                    "Preference Match": "Yes" if preference_pass else "No",
                    "Hours": shift_hours,
                }
            )
            remaining_slots -= 1

    while remaining_slots > 0:
        assignments.append(
            {
                "Day": day,
                "Shift": shift,
                "Employee": "⚠ Unassigned",
                "Tier": "",
                "Preference Match": "",
                "Hours": 0,
            }
        )
        remaining_slots -= 1

    return assignments, employees


def generate_schedule(
    employees: pd.DataFrame,
    selected_days: List[str],
    shift_config: List[Dict],
    shift_hours: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    state = build_employee_state(employees, shift_hours)
    assignments_today: Dict[str, set] = {}

    records: List[Dict] = []
    normalized_shift_names = normalise_shift_config(shift_config)

    for day in selected_days:
        for shift_entry in normalized_shift_names:
            headcount = max(0, shift_entry["Headcount"])
            shift_name = shift_entry["Shift"]
            if headcount == 0:
                continue
            shift_assignments, state = assign_shift(
                state, day, shift_name, headcount, shift_hours, assignments_today
            )
            records.extend(shift_assignments)

    schedule_df = pd.DataFrame(records)
    summary = (
        state[["Name", "Fulltime/Part-time", "Tier", "scheduled_hours", "remaining_hours"]]
        .rename(
            columns={
                "Name": "Employee",
                "Fulltime/Part-time": "Employment Type",
                "scheduled_hours": "Scheduled Hours",
                "remaining_hours": "Hours Left",
            }
        )
        .sort_values(["Tier", "Employee"])
    )

    return schedule_df, summary


def download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    csv = df.to_csv(index=False)
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def main() -> None:
    st.set_page_config(page_title="Shift Scheduler", layout="wide")
    st.title("Shift Scheduler")
    st.caption(
        "Upload or edit your staffing data, tweak shift requirements, and build a schedule "
        "that respects preferences and tiers."
    )

    with st.sidebar:
        st.header("Inputs")
        upload = st.file_uploader("Employee roster (CSV or Excel)", type=["csv", "xls", "xlsx"])
        shift_hours = st.number_input("Hours per shift", min_value=1, max_value=12, value=8, step=1)
        selected_days = st.multiselect(
            "Days to schedule",
            options=DEFAULT_WEEK_DAYS,
            default=DEFAULT_WEEK_DAYS[:5],
        )

        st.markdown("**Shift staffing targets**")
        shift_editor = st.data_editor(
            pd.DataFrame(DEFAULT_SHIFT_CONFIG),
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key="shift_editor",
        )

    raw_df = read_uploaded_file(upload)
    normalized = normalize_columns(raw_df)

    st.subheader("1. Review / edit staffing inputs")
    st.write(
        "Update employee details directly in the table, including availability, shift preference, "
        "tier, and target hours."
    )
    edited_df = st.data_editor(
        normalized,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="employees_editor",
    )

    st.subheader("2. Generate schedule")
    run_schedule = st.button("Generate schedule", type="primary")

    if run_schedule:
        if not selected_days:
            st.error("Please select at least one day to schedule.")
            return

        schedule, summary = generate_schedule(edited_df, selected_days, shift_editor.to_dict("records"), shift_hours)

        if schedule.empty:
            st.warning("No assignments were generated. Please check availability and shift requirements.")
            return

        st.markdown("#### Schedule")
        st.dataframe(schedule, use_container_width=True, hide_index=True)
        download_button(schedule, "Download schedule CSV", "schedule.csv")

        st.markdown("#### Employee summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        download_button(summary, "Download summary CSV", "schedule_summary.csv")

        unassigned = summary[summary["Scheduled Hours"] == 0]
        if not unassigned.empty:
            st.info(
                "Employees without assignments this run: "
                + ", ".join(unassigned["Employee"].tolist())
            )


if __name__ == "__main__":
    main()
