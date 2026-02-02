import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

function getWeekRangeISO(date: Date) {
  // Week starts Monday
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const day = (d.getUTCDay() + 6) % 7; // Mon=0..Sun=6
  const monday = new Date(d);
  monday.setUTCDate(d.getUTCDate() - day);
  const sunday = new Date(monday);
  sunday.setUTCDate(monday.getUTCDate() + 6);

  const toISODate = (x: Date) => x.toISOString().slice(0, 10);
  return { week_start: toISODate(monday), week_end: toISODate(sunday) };
}

serve(async (req) => {
  try {
    const body = await req.json().catch(() => ({}));
    const base = body.week_start ? new Date(body.week_start) : new Date();
    const { week_start, week_end } = getWeekRangeISO(base);

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const serviceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabaseAdmin = createClient(supabaseUrl, serviceRoleKey);

    // Verify manager
    const authHeader = req.headers.get("Authorization") ?? "";
    const jwt = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : null;
    if (!jwt) return new Response(JSON.stringify({ error: "Missing token" }), { status: 401 });

    const supabaseUser = createClient(
      supabaseUrl,
      Deno.env.get("SUPABASE_ANON_KEY")!,
      { global: { headers: { Authorization: `Bearer ${jwt}` } } }
    );

    const { data: userData } = await supabaseUser.auth.getUser();
    const userId = userData?.user?.id;
    if (!userId) return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401 });

    const { data: myProfile } = await supabaseUser.from("profiles").select("role").eq("id", userId).single();
    if (myProfile?.role !== "manager") return new Response(JSON.stringify({ error: "Forbidden" }), { status: 403 });

    // Find active week
    const { data: activeWeek } = await supabaseAdmin
      .from("weeks")
      .select("id")
      .eq("is_active", true)
      .maybeSingle();

    // Delete last week's availability + deactivate old week
    if (activeWeek?.id) {
      await supabaseAdmin.from("availability").delete().eq("week_id", activeWeek.id);
      await supabaseAdmin.from("weeks").update({ is_active: false }).eq("id", activeWeek.id);
    }

    // Create or activate new week
    const { data: existing } = await supabaseAdmin
      .from("weeks")
      .select("id")
      .eq("week_start", week_start)
      .maybeSingle();

    if (existing?.id) {
      await supabaseAdmin.from("weeks").update({ week_end, is_active: true }).eq("id", existing.id);
    } else {
      await supabaseAdmin.from("weeks").insert({ week_start, week_end, is_active: true });
    }

    return new Response(JSON.stringify({ week_start, week_end }), {
      headers: { "Content-Type": "application/json" },
      status: 200,
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), { status: 500 });
  }
});
