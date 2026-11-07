# Kinescript brands: define a brand once, every video follows it

Status: approved 2026-11-07. Step 1 (contracts and core) lands with this file.

## Context

Today a scene can carry a `theme` dict (merged over `components.THEME`, `kinescript/components.py:15`) and use
`"$accent"` anywhere (`scene._subst`, `kinescript/scene.py:1200`). That is per scene only. There is no file shared
between videos, no logo concept, no type styles, chart colours are hard-coded (`CHART_COLORS`, `components.py:128`),
built-ins hard-code sizes/weights/motions and literals (`lowerThird` plate `#0a0c10d0`, shadow `#00000088`),
`canvas.bg` defaults to a literal (`scene.py:223`), and `kine check` knows nothing about a brand.

Goal (Dimitri, 2026-11-07): a user establishes their brand (colours, fonts, charts, logos and logo layout, and more)
once; every video made with it is consistent; a **brand wizard** web UI (like `kine edit`) shows, manages and edits it;
every wizard capability is also an agent capability (MCP + CLI) through the same code.

Inputs: Cod's design review (2026-11-07, gpt-5.6-sol) and Depthcue's shipped brand system
(`Y:\Projects\Depthcue\docs\BRANDS_AND_PPTX.md`). Decisions taken with Dimitri:

- **Library + copy.** Brands live in a per-user library `~/.kinescript/brands/<id>/` (override: `KINE_BRANDS`).
  `kine brand use <id>` copies a snapshot into the project's `brand/` folder and records its digest; running it again
  updates the copy and refuses to overwrite local edits without `--force`. Scenes read only the project copy, so an
  old video renders the same after the library brand changes.
- **Visual first.** v1 = palettes, fonts + type styles, charts, logos + placement + bug, component styles, motion
  defaults, safe area, locks. Intro/outro, audio and grade are later steps in this same plan.

Target release: 1.9.0 (planned from main at 1.8.0, `23718c6`).

## Brand format (`brand/brand.json` + `fonts/` `logos/`)

```json
{
  "kinescriptBrand": 1,
  "id": "acme", "name": "Acme",
  "palettes": {
    "dark":  {"bg": "#0a0c10", "surface": "#242832", "text": "#f2f0e8", "muted": "#9299a8",
              "accent": "#ff633f", "accent2": "#b9f45d", "positive": "#34c759", "negative": "#ff3b30"},
    "light": {"bg": "#f2f0e8", "text": "#0a0c10", "...": "..."}
  },
  "default": "dark",
  "fonts": {
    "display": {"family": "Acme Sans", "files": [{"file": "fonts/AcmeSans-Bold.ttf", "weight": 700}], "license": "OFL"},
    "body": {"family": "Inter"},
    "mono": {"family": "IBM Plex Mono"}
  },
  "text": {
    "title":   {"font": "display", "size": 128, "weight": 800, "tracking": 0},
    "heading": {"font": "display", "size": 72, "weight": 700},
    "body":    {"font": "body", "size": 40, "weight": 400},
    "caption": {"font": "body", "size": 28, "weight": 500, "case": "upper", "tracking": 4},
    "label":   {"font": "mono", "size": 24, "weight": 500}
  },
  "charts": {"series": ["#ff633f", "#b9f45d", "#f2f0e8", "#9299a8", "#5b6272"], "highlight": "accent",
             "axis": {"color": "muted", "width": 3}, "labels": {"text": "caption"}, "format": "{:,.0f}"},
  "logos": {
    "full": {"file": "logos/acme.svg", "onDark": "logos/acme-light.svg", "alt": "Acme"},
    "mark": {"file": "logos/mark.svg"},
    "clearSpace": 0.25, "minHeight": 48
  },
  "placements": {"corner": {"anchor": "bottomright", "margin": "4%", "height": "6%"}},
  "bug": {"logo": "mark", "placement": "corner", "opacity": 0.85},
  "safeArea": "5%",
  "motion": {"enter": "rise", "exit": "fade", "ease": "outCubic", "dur": 0.5},
  "components": {"lowerThird": {"plate": "surface", "bar": "accent", "titleText": "heading", "subText": "body"}},
  "lock": ["palettes", "fonts"]
}
```

Rules:

- **Scene use:** top-level `"brand": "brand"` (a folder path relative to the scene, normally the project's `brand/`)
  and optional `"variant": "light"`. Video has no viewer OS theme, so the variant is explicit (default from the brand).
- **Resolution order:** built-in `THEME` < brand (selected palette, flattened) < `scene.theme`, unless `lock` names
  that part, in which case the scene override is an error.
- **Names stay `$name`.** The selected palette flattens into the existing theme names, so `"$accent"`, `"$text"`,
  `"$surface"` work everywhere and old scenes are untouched; the old names (`ink`, `paper`, `muted`, `void`...) are
  aliases filled from palette roles. Dotted names reach the rest (`"$charts.series"`, `"$logo.mark"`). Cod suggested
  typed `{"$token": ...}` objects; rejected because it breaks every existing scene and makes JSON heavier for agents.
  Dependency discovery scans strings instead.
- **Paths:** every file in a brand resolves against the brand folder, at load, to an absolute path; `..`, absolute
  paths and symlinks leaving the folder are refused.
- **Text styles:** a text/counter layer takes `"textStyle": "heading"` (font, size, weight, tracking, case); the
  layer's own keys win. (`"style"` is already taken in `scene.py:25`; `"role"` is taken by sounds.)
- **Component styles** are a bounded, versioned set per built-in: colours (palette role names), text styles, plate,
  stroke, radius, and a named motion. Not arbitrary default parameters, so a brand cannot silently change timing.
- **Clear space** is a fraction of the logo's own box, not x-height (many marks have no type).
- **Determinism:** the brand is loaded, validated and flattened once in `Scene.__init__`, before `_subst`. Frames read
  only the resolved scene. The brand digest (sha256 over `brand.json` and every referenced file) joins any cache key
  that keys on the scene (check `output.py:637` caches).

## Code layout

- **New `kinescript/brand.py`** (core, no HTTP): `load(folder) -> Brand`, `resolve(brand, variant) -> theme dict`,
  `digest(folder)`, `validate(folder) -> issues`, and the operations every surface calls:
  `get`, `set(path, value, base_rev)` (JSON-pointer path), `add_file(kind, name, bytes, base_rev, replace)`,
  `remove_file`, `init(id)`, `use(id, project, force)`, `list_library()`, `preview(folder, what) -> png`.
  Revision = digest; a write with a stale `base_rev` is refused (same pattern as `editor.set_props` `base_hash`,
  `kinescript/editor.py:120`). Writes go to a temp file then `os.replace`, files content-checked (TTF/OTF/TTC magic,
  PNG/JPEG magic, SVG parsed with the existing `svgdoc.py` and stripped of scripts/foreign refs), size-capped.
- **`kinescript/scene.py`**: read `brand`/`variant`, merge per the order above, `canvas.bg` default becomes `$bg`.
- **`kinescript/components.py`**: built-ins read `th["charts"]`, text styles and component styles; `CHART_COLORS`
  becomes the fallback when there is no brand; remove literal plate/shadow colours (derive from palette).
  New built-ins: `logo` (`{"use": "logo", "variant": "full", "placement": "corner"}`) and `bug` (a whole-video logo).
- **`kinescript/render.py` `_typeface` (`:2969`)**: brand faces register by family name -> file, so `"font": "Acme
  Sans"` and the theme font find the brand's file before system lookup.
- **`kinescript/schema.py`**: `brand`, `variant`, `textStyle` keys; brand file schema; `kine docs` section in
  `kinescript/docs.py`.
- **`kinescript/verify.py` `check` (`:648`)**: brand issues (below).
- **`kinescript/brand_ui/index.html`** + **`kinescript/brand_server.py`**: the wizard; reuses the editor's handler
  pattern (Host guard `_host_name`/`LOCAL_HOSTS`, `MAX_BODY`, JSON-only POST, `_guard`) — factor those helpers out
  of `editor.py` into a small shared module rather than copying them. Package data in `pyproject.toml`.
- **`kinescript/mcp.py`**: one tool `kine_brand` with `action` = `list|show|init|use|set|add_file|remove_file|check|
  preview`, each a thin call into `brand.py`. **`kinescript/cli.py`**: `kine brand init|list|show|use|set|check|
  preview|wizard`.

## Checks

On by default:
- error: brand file invalid or of a newer version, missing or unsafe file, digest mismatch with the recorded copy
  (warning, with "run kine brand use" fix), unknown `$name`/`textStyle`/placement, locked part overridden, palette
  missing a required role, a declared text/bg pair under 4.5:1 (3:1 for title sizes), chart series under 3:1 vs bg
  (WCAG 1.4.11), font weight asked that no file has and no fallback.
- warning: font without license metadata, logo drawn below `minHeight` or inside its clear space with other layers,
  bug overlapping text, fallback font used.

Opt-in with `kine check --brand-strict` (Cod: off by default to avoid lint fatigue on deliberate exceptions):
off-palette colour (nearest brand colour and its delta-E in the fix), non-brand font, text outside the brand safe
area, logo off its placement.

## Brand wizard (`kine brand wizard [id|folder]`)

127.0.0.1 only, same guard as `kine edit`. Panels, each backed by the `brand.py` operations above (the HTTP routes
and `kine_brand` call the same functions, and the same edit through either writes the same bytes):

1. **Library:** list brands, new from the default Kinescript brand, duplicate, open, "use in project".
2. **Colours:** both palettes as swatches with role names, a contrast matrix of the declared pairs with refusals shown
   at the swatch, chart series row with the 3:1 check.
3. **Type:** upload font files (or name an installed family), license field, the text styles with live samples drawn
   by the real renderer.
4. **Logos:** upload variants (light/dark/mark), alt text, clear space and min size, placement presets dragged on a
   sample frame.
5. **Components and charts:** a gallery — lowerThird, titleCard, callout, barChart, lineChart, pieChart, logo, bug —
   rendered in the brand via `brand.preview`, which renders a small built-in sample scene with the real renderer.
6. **Raw JSON** view with the validation result, and a "what's inherited / overridden / locked" view for a project
   copy versus its library brand.

Changes save immediately with `base_rev`; a stale save shows the refusal and reloads.

## Steps (one PR each)

1. **Contracts + core.** `brand.py` load/resolve/digest/validate, scene `brand`/`variant`, theme flattening and
   aliases, path safety, schema + docs, `docs/PLAN-1.9-BRAND.md` (this plan) lands here. Test: a scene with no brand
   renders byte-identical frames to 1.8.0 on a set of example scenes; brand scene resolves `$accent`/`$text`; locked
   override errors; traversal refused; `test_frame_is_pure_function_of_index` with a brand.
2. **Brand drives the built-ins.** Charts from `charts`, text styles (`textStyle`), component styles, motion
   defaults, `canvas.bg` from `$bg`, brand fonts in `_typeface`. Test: each built-in rendered under two different
   brands shows each brand's colours/fonts (pixel samples); no-brand output unchanged.
3. **Logos.** `logo` and `bug` components, placements, clear space/min size, SVG/PNG logos. Test: placement boxes via
   `kine probe`, `kine look --boxes` on a branded sample.
4. **Checks.** Default and `--brand-strict` rules above. Test: one fixture per rule, plus "a clean branded example has
   zero brand issues".
5. **Library and ops.** `~/.kinescript/brands`, `init|list|show|use|set|add_file|remove_file|preview`, digest
   recording, atomic writes, upload sniffing, CLI + `kine_brand` MCP tool. A default "Kinescript" brand (from
   `brand/BRAND.md`) and a second made-up fixture brand with OFL fonts and its own logo. Test: CLI and MCP produce
   identical files for the same op; stale `base_rev` refused; `use` refuses over local edits.
6. **Wizard.** `brand_server.py` + `brand_ui/index.html` on step 5's ops; shared HTTP helpers factored out of
   `editor.py`. Test: HTTP route tests like the editor's, plus a real-browser pass (Claude in Chrome) over every
   panel with screenshots viewed.
7. **Intro/outro.** Brand `sting` (an intro component) and `endCard`, placed by `"intro": true`/`"outro": true` on a
   scene or a sequence. Test: timing via probe, look.
8. **Audio defaults.** Sonic logo sound, music bed gain, voice backend/speaker defaults, all overridable per sound.
   Test: check's audio lints on a branded scene.
9. **Grade + import.** Brand `grade` defaults (the existing `grade` keys; a LUT file only if `grade` already takes
   one), and `kine brand import <logo.svg|png>` that proposes a palette from the logo for the wizard to confirm.
   Website CSS import is out of scope (URL safety, ambiguous mapping).
10. **Close-out.** README section, site demo in two brands, the agent skill (`making-kinescript-videos`, PR #126)
    gets a brand section, changelog, 1.9.0.

Steps 1–6 make v1; 7–9 can each ship after it. Cod reviews the step 1 contracts PR.

## Verification (every step)

- `.venv/Scripts/python -m pytest` green; new tests per step as listed.
- Visual changes: `kine check`, then `kine look --boxes` on `examples/brand.json` (new) and look at the image, per
  CLAUDE.md.
- Regression: render a fixed set of existing examples before and after; frames must match (no brand = no change).
- MCP parity: run each `kine_brand` action through `kinescript.mcp.call` in a test and compare with the CLI result.
- Wizard: open `kine brand wizard` in Chrome, exercise each panel, confirm the files on disk match what an MCP call
  writes for the same change.

## Not in this plan

Live-linked brands that change old videos, website CSS import, `.potx`/`.pptx` import, font subsetting, per-viewer
light/dark switching, a hosted/shared brand registry.
