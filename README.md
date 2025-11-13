# Scheduling-app
## Web App
Interactive scheduler that lets you upload or edit employee rosters, define shift staffing targets, and auto-build a plan that respects availability, shift preferences, and tier priorities.
### Quick start
```
pip install -r requirements.txt
streamlit run app.py
```
Open the link Streamlit prints (typically `http://localhost:8501`) and:
- Upload a CSV/Excel roster or edit the sample inline.
- Adjust days, shift lengths, and required headcount per shift.
- Click **Generate schedule** to produce a downloadable plan and per-employee summary.
The app expects these columns (names are case-insensitive and auto-normalised):
- `Name`
- `Total Hours` (weekly target)
- `Available Days` (comma-separated list, e.g. `Mon,Wed,Fri`)
- `Shift Preference` (e.g. `Morning`, `Evening`, `Night`)
- `Fulltime/Part-time`
- `Tier` (lower numbers = higher priority)
Missing columns are filled with sensible defaults so you can start from scratch if needed. Adjust rows directly in the table and re-run the scheduler as plans evolve.

## Docker
Build the image:
```
docker build -t scheduler-app .
```
Run it:
```
docker run -p 8501:8501 scheduler-app
```
The app will be available at `http://localhost:8501`. Mount a volume or bake updated CSV/XLSX files into the image if you need to work with non-sample data.