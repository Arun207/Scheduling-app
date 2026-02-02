export function $(id){ return document.getElementById(id); }

export function setText(id, text){
  const el = $(id);
  if (el) el.textContent = text ?? "";
}

export function show(id, visible){
  const el = $(id);
  if (!el) return;
  el.classList.toggle("hidden", !visible);
}

export function escapeHtml(str){
  return String(str ?? "")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}
