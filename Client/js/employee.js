import { supabase } from "./supabaseClient.js";

export const DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];

export function renderDays(daysBoxEl){
  daysBoxEl.innerHTML = "";
  for (const d of DAYS){
    const wrap = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = d;
    wrap.appendChild(cb);
    wrap.appendChild(document.createTextNode(d));
    daysBoxEl.appendChild(wrap);
  }
}

export function getSelectedDays(daysBoxEl){
  const cbs = daysBoxEl.querySelectorAll("input[type=checkbox]");
  return [...cbs].filter(x => x.checked).map(x => x.value);
}

export function setSelectedDays(daysBoxEl, days){
  const set = new Set(days || []);
  const cbs = daysBoxEl.querySelectorAll("input[type=checkbox]");
  for (const cb of cbs) cb.checked = set.has(cb.value);
}

export async function getMyProfile(){
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;
  const { data, error } = await supabase.from("profiles").select("*").eq("id", user.id).single();
  if (error) return null;
  return data;
}

export async function getActiveWeek(){
  const { data, error } = await supabase.from("weeks").select("*").eq("is_active", true).maybeSingle();
  if (error) throw error;
  return data;
}

export async function loadMyAvailability(weekId, userId){
  const { data, error } = await supabase
    .from("availability")
    .select("*")
    .eq("week_id", weekId)
    .eq("user_id", userId)
    .maybeSingle();
  if (error) throw error;
  return data;
}

export async function saveMyAvailability(payload){
  const { error } = await supabase
    .from("availability")
    .upsert(payload, { onConflict: "week_id,user_id" });
  if (error) throw error;
}
