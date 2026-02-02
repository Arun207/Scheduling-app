import { supabase } from "./supabaseClient.js";

export async function login(email, password){
  const { data, error } = await supabase.auth.signInWithPassword({ email, password });
  if (error) throw error;
  return data;
}

export async function logout(){
  const { error } = await supabase.auth.signOut();
  if (error) throw error;
}

export async function getSession(){
  return await supabase.auth.getSession();
}

export async function getUser(){
  return await supabase.auth.getUser();
}

export function onAuthChange(cb){
  supabase.auth.onAuthStateChange((_event, session) => cb(session));
}
