export function exportToExcel(rows, fileName = "weekly_schedule.xlsx"){
  // rows should already have the 4 columns you want
  const sheetData = [
    ["Name", "Available Days", "Shift Preference", "Fulltime/Part-time"],
    ...rows.map(r => [
      r.name,
      (r.available_days || []).join(", "),
      r.shift_preference,
      r.employment_type === "full_time" ? "Full-time" : "Part-time"
    ])
  ];

  const wb = XLSX.utils.book_new();
  const ws = XLSX.utils.aoa_to_sheet(sheetData);
  XLSX.utils.book_append_sheet(wb, ws, "Schedule");
  XLSX.writeFile(wb, fileName);
}
