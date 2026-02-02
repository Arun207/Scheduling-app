import { supabase } from "./supabaseClient.js";
import { escapeHtml } from "./ui.js";

/* ---------------------------
   Create Employee (Manager)
   Uses supabase.functions.invoke (recommended)
---------------------------- */
export async function createEmployee(email, full_name) {
  const { data, error } = await supabase.functions.invoke("create-employee", {
    body: { email, full_name }
  });

  if (error) {
    // error.message is usually the useful one
    throw new Error(error.message || "Create employee failed");
  }

  // expected: { email, tempPassword }
  return data;
}

/* ---------------------------
   Start New Week (Manager)
---------------------------- */
export async function startNewWeek() {
  const { data, error } = await supabase.functions.invoke("start-new-week", {
    body: {}
  });

  if (error) {
    throw new Error(error.message || "Start new week failed");
  }

  return data; // { week_start, week_end, ... }
}

/* ---------------------------
   Get Active Week (shared)
---------------------------- */
export async function getActiveWeek() {
  const { data, error } = await supabase
    .from("weeks")
    .select("*")
    .eq("is_active", true)
    .maybeSingle();

  if (error) throw error;
  return data;
}

/* ---------------------------
   Load All Availability (Manager)
---------------------------- */
export async function loadWeekAvailability(weekId) {
  const { data, error } = await supabase
    .from("availability")
    .select(`
      name,
      available_days,
      shift_preference,
      employment_type,
      updated_at
    `)
    .eq("week_id", weekId)
    .order("updated_at", { ascending: false });

  if (error) throw error;
  return data || [];
}

/* ---------------------------
   Render Manager Table
---------------------------- */
export function renderManagerTable(tbodyEl, rows) {
  if (!tbodyEl) return;
  tbodyEl.innerHTML = "";

  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(r.name)}</td>
      <td>${escapeHtml((r.available_days || []).join(", "))}</td>
      <td>${escapeHtml(r.shift_preference || "")}</td>
      <td>${r.employment_type === "full_time" ? "Full-time" : "Part-time"}</td>
      <td>${new Date(r.updated_at).toLocaleString()}</td>
    `;
    tbodyEl.appendChild(tr);
  }
}
