import { $, setText, show } from "./ui.js";
import { login, logout, onAuthChange, getSession } from "./auth.js";
import {
  renderDays, getSelectedDays, setSelectedDays,
  getMyProfile, getActiveWeek, loadMyAvailability, saveMyAvailability
} from "./employee.js";
import {
  createEmployee, startNewWeek,
  getActiveWeek as getActiveWeekManager,
  loadWeekAvailability, renderManagerTable
} from "./manager.js";
import { exportToExcel } from "./excel.js";
import { supabase } from "./supabaseClient.js";

// ---------- Helpers ----------
function safeOnClick(id, handler) {
  const el = $(id);
  if (!el) return;
  el.addEventListener("click", handler);
}

function togglePassword(btnId, inputId) {
  const btn = $(btnId);
  const input = $(inputId);
  if (!btn || !input) return;

  btn.addEventListener("click", () => {
    const isHidden = input.type === "password";
    input.type = isHidden ? "text" : "password";
    btn.textContent = isHidden ? "Hide" : "Show";
    input.focus();
  });
}

// ---------- Initial UI setup ----------
const daysBox = $("daysBox");
if (daysBox) renderDays(daysBox);

// ---------- Core UI ----------
async function updateUI() {
  try {
    const { data: { session } } = await getSession();
    console.log("SESSION:", session);

    const loggedIn = !!session?.user;

    setText("authStatus", loggedIn ? `Logged in as ${session.user.email}` : "Not logged in");
    show("btnLogout", loggedIn);
    show("authCard", !loggedIn);

    // Default hide both dashboards
    show("employeeCard", false);
    show("managerCard", false);

    if (!loggedIn) return;

    // Helpful message so it doesn't feel "stuck"
    setText("authMsg", "Loading dashboard...");

    const profile = await getMyProfile();
    console.log("PROFILE:", profile);

    if (!profile) {
      setText("authMsg", "Profile missing. Manager must create your profile in the profiles table.");
      return;
    }

    const activeWeek = await getActiveWeek();
    console.log("ACTIVE WEEK:", activeWeek);

    if (!activeWeek) {
      setText("authMsg", "No active week. Manager must create/start a week.");
      return;
    }

    const weekText = `Active week: ${activeWeek.week_start} → ${activeWeek.week_end}`;
    setText("weekInfo", weekText);
    setText("managerWeekInfo", weekText);

    // Clear auth message after successful load
    setText("authMsg", "");

    if (profile.role === "manager") {
      show("managerCard", true);

      const tbody = $("mgrTableBody");
      if (tbody) {
        const rows = await loadWeekAvailability(activeWeek.id);
        renderManagerTable(tbody, rows);
        setText("mgrMsg", `Loaded ${rows.length} submission(s).`);
      }
    } else {
      show("employeeCard", true);

      const my = await loadMyAvailability(activeWeek.id, profile.id);

      if ($("empName")) $("empName").value = (my?.name || profile.full_name || "");
      if (daysBox) setSelectedDays(daysBox, my?.available_days || []);
      if ($("shiftPref")) $("shiftPref").value = (my?.shift_preference || "Any");
      if ($("employmentType")) $("employmentType").value = (my?.employment_type || "part_time");

      setText("empMsg", my ? "Loaded your saved availability." : "No availability submitted yet for this week.");
    }
  } catch (err) {
    console.error("updateUI crashed:", err);
    setText("authMsg", `updateUI error: ${err?.message || String(err)}`);
  }
}

// ---------- Auth actions ----------
safeOnClick("btnLogin", async () => {
  const btn = $("btnLogin");
  if (btn) btn.disabled = true;

  try {
    const email = $("loginEmail")?.value?.trim() || "";
    const password = $("loginPassword")?.value || "";

    console.log("Attempt login:", email);
    setText("authMsg", "Logging in...");

    const result = await login(email, password);
    console.log("LOGIN RESULT:", result);

    // Force UI update immediately
    await updateUI();
  } catch (e) {
    console.error("Login failed:", e);
    setText("authMsg", e?.message || String(e));
  } finally {
    if (btn) btn.disabled = false;
  }
});

safeOnClick("btnLogout", async () => {
  try {
    await logout();
    // UI will update via auth change, but do it now too
    await updateUI();
  } catch (e) {
    console.error("Logout failed:", e);
  }
});

// ---------- Employee actions ----------
safeOnClick("btnSaveAvailability", async () => {
  try {
    const profile = await getMyProfile();
    const week = await getActiveWeek();
    if (!profile || !week) throw new Error("Missing profile or active week");

    const name = $("empName")?.value?.trim() || "";
    const days = daysBox ? getSelectedDays(daysBox) : [];

    if (!name) throw new Error("Name is required");
    if (days.length === 0) throw new Error("Select at least one available day");

    await saveMyAvailability({
      week_id: week.id,
      user_id: profile.id,
      name,
      available_days: days,
      shift_preference: $("shiftPref")?.value || "Any",
      employment_type: $("employmentType")?.value || "part_time",
    });

    setText("empMsg", "Saved!");
  } catch (e) {
    setText("empMsg", e?.message || String(e));
  }
});

// Show change password box
safeOnClick("btnChangePassword", () => {
  const box = $("changePasswordBox");
  if (box) box.classList.remove("hidden");

  if ($("changePasswordMsg")) $("changePasswordMsg").textContent = "";
  if ($("newPassword")) $("newPassword").value = "";
  if ($("confirmPassword")) $("confirmPassword").value = "";
});

safeOnClick("btnSaveNewPassword", async () => {
  try {
    const pwd = $("newPassword")?.value || "";
    const confirm = $("confirmPassword")?.value || "";

    if (pwd.length < 6) throw new Error("Password must be at least 6 characters");
    if (pwd !== confirm) throw new Error("Passwords do not match");

    const { error } = await supabase.auth.updateUser({ password: pwd });
    if (error) throw error;

    $("changePasswordBox")?.classList.add("hidden");
    setText("empMsg", "Password updated successfully.");
  } catch (e) {
    if ($("changePasswordMsg")) $("changePasswordMsg").textContent = e?.message || String(e);
  }
});

safeOnClick("btnCancelPassword", () => {
  $("changePasswordBox")?.classList.add("hidden");
});

// ---------- Manager actions ----------
safeOnClick("btnCreateEmployee", async () => {
  try {
    const email = $("newEmpEmail")?.value?.trim() || "";
    const full_name = $("newEmpName")?.value?.trim() || "";
    if (!email || !full_name) throw new Error("Email and name required");

    setText("createEmpMsg", "Creating employee...");
    const result = await createEmployee(email, full_name);

    // Show temp password in message (you can replace with your password reveal UI if you want)
    setText("createEmpMsg", `Created: ${result.email} | Temp password: ${result.tempPassword}`);
  } catch (e) {
    setText("createEmpMsg", e?.message || String(e));
  }
});

safeOnClick("btnStartNewWeek", async () => {
  try {
    setText("mgrMsg", "Starting new week...");
    const res = await startNewWeek();
    setText("mgrMsg", `New active week: ${res.week_start} → ${res.week_end} (last week deleted)`);
    await updateUI();
  } catch (e) {
    setText("mgrMsg", e?.message || String(e));
  }
});

safeOnClick("btnRefreshTable", async () => {
  try {
    const week = await getActiveWeekManager();
    if (!week) throw new Error("No active week");

    const rows = await loadWeekAvailability(week.id);
    renderManagerTable($("mgrTableBody"), rows);
    setText("mgrMsg", `Refreshed. ${rows.length} submission(s).`);
  } catch (e) {
    setText("mgrMsg", e?.message || String(e));
  }
});

safeOnClick("btnExportExcel", async () => {
  try {
    const week = await getActiveWeekManager();
    if (!week) throw new Error("No active week");

    const rows = await loadWeekAvailability(week.id);
    exportToExcel(rows, `schedule_${week.week_start}.xlsx`);
    setText("mgrMsg", "Excel exported.");
  } catch (e) {
    setText("mgrMsg", e?.message || String(e));
  }
});

// ---------- Password toggles ----------
togglePassword("btnToggleLoginPassword", "loginPassword");
togglePassword("btnToggleNewPassword", "newPassword");
togglePassword("btnToggleConfirmPassword", "confirmPassword");

// ---------- Auth listener (fires on login/logout/refresh) ----------
onAuthChange((session) => {
  console.log("Auth state changed:", session);
  updateUI();
});

// Initial load
updateUI();
