"""SkSL shader layers. Built-in uniforms: iTime (layer seconds), iResolution (w, h), iProgress (0..1).
Colour uniforms are float4 (premultiply is not needed; alpha stays 1 unless you set it)."""
from __future__ import annotations

PRESETS = {
    # Four soft colour fields drifting over each other.
    "mesh": ("""
uniform float iTime; uniform float2 iResolution; uniform float iProgress;
uniform float4 c1; uniform float4 c2; uniform float4 c3; uniform float4 c4; uniform float speed;
half4 main(float2 p) {
  float2 uv = p / iResolution; float t = iTime * speed;
  float2 a = float2(0.25 + 0.15 * sin(t * 0.7), 0.3 + 0.2 * cos(t * 0.5));
  float2 b = float2(0.75 + 0.15 * cos(t * 0.6), 0.25 + 0.15 * sin(t * 0.8));
  float2 c = float2(0.3 + 0.2 * cos(t * 0.4), 0.8 + 0.1 * sin(t * 0.9));
  float2 d = float2(0.8 + 0.1 * sin(t * 0.5), 0.75 + 0.15 * cos(t * 0.3));
  float wa = 1.0 / (0.02 + pow(distance(uv, a), 2.0)); float wb = 1.0 / (0.02 + pow(distance(uv, b), 2.0));
  float wc = 1.0 / (0.02 + pow(distance(uv, c), 2.0)); float wd = 1.0 / (0.02 + pow(distance(uv, d), 2.0));
  float4 col = (c1 * wa + c2 * wb + c3 * wc + c4 * wd) / (wa + wb + wc + wd);
  return half4(col.rgb, 1.0);
}""", {"c1": "#ff633f", "c2": "#b9f45d", "c3": "#242832", "c4": "#0a0c10", "speed": 1.0}),
    # Horizontal light bands with a slow vertical sway.
    "aurora": ("""
uniform float iTime; uniform float2 iResolution; uniform float iProgress;
uniform float4 base; uniform float4 glow; uniform float4 glow2; uniform float speed;
half4 main(float2 p) {
  float2 uv = p / iResolution; float t = iTime * speed;
  float band = 0.0;
  for (int i = 0; i < 3; i++) {
    float fi = float(i);
    float y = 0.35 + 0.12 * fi + 0.08 * sin(uv.x * (3.0 + fi) + t * (0.6 + 0.2 * fi));
    band += exp(-pow((uv.y - y) * (9.0 - fi * 2.0), 2.0)) * (0.7 - 0.15 * fi);
  }
  float4 g = mix(glow, glow2, 0.5 + 0.5 * sin(uv.x * 2.0 + t * 0.3));
  float4 col = mix(base, g, clamp(band * 0.75, 0.0, 0.85));
  return half4(col.rgb, 1.0);
}""", {"base": "#0a0c10", "glow": "#b9f45d", "glow2": "#ff633f", "speed": 1.0}),
    # Receding floor grid, like a stage.
    "grid": ("""
uniform float iTime; uniform float2 iResolution; uniform float iProgress;
uniform float4 base; uniform float4 line; uniform float speed; uniform float density;
half4 main(float2 p) {
  float2 uv = p / iResolution;
  float horizon = 0.42;
  float4 col = base;
  if (uv.y > horizon) {
    float z = 1.0 / (uv.y - horizon + 0.001);
    float gx = (uv.x - 0.5) * z * density;
    float gz = z * density * 0.5 + iTime * speed;
    float lx = abs(fract(gx) - 0.5); float lz = abs(fract(gz) - 0.5);
    float w = 0.004 * z;
    float g = max(smoothstep(w + 0.012, w, lx), smoothstep(w + 0.012, w, lz));
    float fade = clamp((uv.y - horizon) * 2.4, 0.0, 1.0);
    col = mix(base, line, g * fade);
  }
  return half4(col.rgb, 1.0);
}""", {"base": "#0a0c10", "line": "#ff633f", "speed": 0.6, "density": 1.0}),
    # Timing grid: beat lines, bar lines every fourth beat, faint lanes, registration crosses.
    # It advances one beat at a time (discrete steps), never drifts.
    "score": ("""
uniform float iTime; uniform float2 iResolution; uniform float iProgress;
uniform float4 base; uniform float4 line; uniform float4 mark; uniform float beat; uniform float rate;
uniform float lane;
half4 main(float2 p) {
  float shift = floor(iTime * rate) * beat;
  float x = p.x + shift;
  float bx = abs(fract(x / beat) - 0.5) * beat;
  float barx = abs(fract(x / (beat * 4.0)) - 0.5) * beat * 4.0;
  float ly = abs(fract(p.y / lane) - 0.5) * lane;
  float beatLine = smoothstep(1.2, 0.2, bx) * 0.35;
  float barLine = smoothstep(1.4, 0.3, barx) * 0.8;
  float laneLine = smoothstep(1.0, 0.2, ly) * 0.25;
  float cross = 0.0;
  if (barx < 9.0 && ly < 9.0) { cross = max(smoothstep(1.3, 0.3, barx), smoothstep(1.3, 0.3, ly)); }
  float4 col = base;
  col = mix(col, line, clamp(max(max(beatLine, barLine), laneLine), 0.0, 1.0));
  col = mix(col, mark, cross * 0.9);
  return half4(col.rgb, 1.0);
}""", {"base": "#0a0c10", "line": "#242832", "mark": "#9299a8", "beat": 48.0, "rate": 2.0, "lane": 216.0}),
}
