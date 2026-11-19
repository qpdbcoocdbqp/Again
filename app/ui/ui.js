// js_on_load for the gr.HTML in sdg_app.py. All editing happens in the browser (materials are edited live on the
// loaded model, the GLB is never rebuilt); Python is only asked to randomize, validate and read/write YAML.
const $ = (s) => element.querySelector(s);
const $$ = (s) => [...element.querySelectorAll(s)];
const CONTENTS = ["off", "solid", "color_bars", "noise"];
const RANGES = [["roughness", 0, 1, 0.01], ["metallic", 0, 1, 0.01], ["exposure", 0.01, 5, 0.05],
                ["yaw", -360, 360, 1], ["pitch", -360, 360, 1], ["roll", -360, 360, 1]];

let D = null;          // init payload from Python
let R = null;          // current recipe (same structure as the YAML)
let viewer = null, mats = [], sel = -1, mask = false, texCache = {}, applyToken = 0;

// ---- color helpers: recipe colors are sRGB 0..1 (what the picker shows); glTF factors are linear
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const lin = (c) => c.map((v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)));
const toHex = (c) => "#" + c.map((v) => Math.round(clamp01(v) * 255).toString(16).padStart(2, "0")).join("");
const fromHex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
function partColor(i) {  // distinct color per part for the segmentation preview
  const h = (i * 0.618034) % 1, s = 0.75, v = 0.95;
  const f = (n) => { const k = (n + h * 6) % 6; return v - v * s * Math.max(0, Math.min(k, 4 - k, 1)); };
  return [f(5), f(3), f(1)];
}
const mulberry32 = (a) => () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

function showErrors(errs) {
  const box = $(".sdg-err");
  box.hidden = !errs || !errs.length;
  box.textContent = (errs || []).join("\n");
}

// ---- screen content textures
async function screenTexture(kind) {
  const key = kind === "noise" ? "noise:" + R.seed : kind;
  if (texCache[key]) return texCache[key];
  const c = document.createElement("canvas");
  c.width = 256; c.height = 144;
  const g = c.getContext("2d");
  if (kind === "color_bars") {
    const cols = ["#c0c0c0", "#c0c000", "#00c0c0", "#00c000", "#c000c0", "#c00000", "#0000c0"];
    cols.forEach((col, i) => { g.fillStyle = col; g.fillRect(Math.floor(i * c.width / 7), 0, Math.ceil(c.width / 7), c.height * 0.75); });
    const grad = g.createLinearGradient(0, 0, c.width, 0);
    grad.addColorStop(0, "#000"); grad.addColorStop(1, "#fff");
    g.fillStyle = grad; g.fillRect(0, c.height * 0.75, c.width, c.height * 0.25);
  } else {
    const rnd = mulberry32(R.seed), img = g.createImageData(c.width, c.height);
    for (let i = 0; i < img.data.length; i += 4) { const v = rnd() * 255; img.data.set([v, v, v, 255], i); }
    g.putImageData(img, 0, 0);
  }
  return (texCache[key] = await viewer.createTexture(c.toDataURL()));
}

// ---- push the recipe to the 3D model
async function apply() {
  try { await applyNow(); } catch (err) { showErrors(["render failed: " + err]); console.error(err); }
}

async function applyNow() {
  if (!viewer || !viewer.model || !mats.length) return;
  const token = ++applyToken;
  const scr = R.screen.part ? D.parts.findIndex((p) => p.name === R.screen.part) : -1;
  const tex = scr >= 0 && !mask && (R.screen.content === "color_bars" || R.screen.content === "noise")
    ? await screenTexture(R.screen.content) : null;
  if (token !== applyToken) return;  // a newer apply started while the texture loaded

  D.parts.forEach((p, i) => {
    const m = mats[i], r = R.parts[p.name];
    if (!m || !r) return;
    const pbr = m.pbrMetallicRoughness;
    let base = lin(r.base_color), emi = [0, 0, 0], metal = r.metallic, rough = r.roughness;
    if (mask) { base = [0, 0, 0]; emi = partColor(i); metal = 0; rough = 1; }
    else if (i === scr && R.screen.content !== "off") {
      base = [0, 0, 0];
      emi = R.screen.content === "solid" ? lin(R.screen.color) : [1, 1, 1];
    } else if (i === sel) emi = [0.3, 0.25, 0];
    m.setAlphaMode(r.visible ? "OPAQUE" : "BLEND");
    pbr.setBaseColorFactor([...base, r.visible ? 1 : 0]);
    pbr.setMetallicFactor(metal);
    pbr.setRoughnessFactor(rough);
    m.setEmissiveFactor(emi);
    if (i === scr) {
      pbr.baseColorTexture.setTexture(null);
      m.emissiveTexture.setTexture(tex);
    }
  });

  const s = R.scene;
  viewer.exposure = mask ? 1 : s.exposure;
  viewer.setAttribute("environment-image", s.environment);
  viewer.setAttribute("shadow-intensity", s.ground ? "1" : "0");
  viewer.orientation = `${R.pose.roll}deg ${R.pose.pitch}deg ${R.pose.yaw}deg`;
  const bg = s.background;
  $(".sdg-view").style.background = bg.type === "solid" ? bg.colors[0] : `linear-gradient(${bg.colors[0]}, ${bg.colors[1]})`;
}

// ---- recipe -> controls
function buildStatic() {
  $(".f-screen").replaceChildren(...CONTENTS.map((c) => new Option(c, c)));
  const rnd = $(".rnd");
  const rows = RANGES.map(([k, lo, hi, step]) => `<span>${k}</span><span class="sdg-row rng" data-k="${k}">
      <input type="number" data-r="${k}" data-i="0" min="${lo}" max="${hi}" step="${step}">
      <span>to</span><input type="number" data-r="${k}" data-i="1" min="${lo}" max="${hi}" step="${step}"></span>`);
  rows.push(`<span>hue jitter</span><input type="number" data-j="hue_jitter" min="0" max="1" step="0.01">`,
            `<span>value jitter</span><input type="number" data-j="value_jitter" min="0" max="1" step="0.01">`,
            `<span>screen content</span><span class="chk">${CONTENTS.map((c) => `<label><input type="checkbox" data-sc="${c}"> ${c}</label>`).join("")}</span>`,
            `<span>background</span><label><input type="checkbox" class="f-rbg"> also randomize</label>`);
  rnd.innerHTML = rows.join("");
  $(".sdg-parts").replaceChildren(...D.parts.map((p, i) => {
    const row = document.createElement("div");
    row.className = "sdg-part"; row.dataset.i = i;
    row.innerHTML = `<input type="checkbox" class="vis" title="show / hide"><input type="color" class="col">
      <span class="nm"></span>
      <div class="sl"><span>rgh</span><input type="range" class="rg" min="0" max="1" step="0.01"><output></output></div>
      <div class="sl"><span>met</span><input type="range" class="mt" min="0" max="1" step="0.01"><output></output></div>`;
    const nm = row.querySelector(".nm");
    nm.textContent = p.name.replace(/^m_opaque__/, "");
    nm.title = `${p.name}\n${p.faces.toLocaleString()} faces, ${p.tris.toLocaleString()} triangles` + (p.screen ? "\n(detected as the screen)" : "");
    if (p.screen) nm.insertAdjacentHTML("beforeend", "<small>screen</small>");
    return row;
  }));
  const [x, y, z] = D.size_m;
  $(".sdg-size").textContent = `Real size: x ${x} · y ${y} · z ${z} m  (width ${y}, height ${z}, depth ${x})`;
  $(".sdg-part-name").textContent = R.screen.part ? "— " + R.screen.part : "— no screen part found";
  $(".f-screen").disabled = $(".f-screen-color").disabled = !R.screen.part;
}

function syncControls() {
  $(".f-name").value = R.name;
  $(".f-seed").value = R.seed;
  $$(".sdg-part").forEach((row) => {
    const i = +row.dataset.i, r = R.parts[D.parts[i].name];
    row.classList.toggle("sel", i === sel);
    row.querySelector(".vis").checked = r.visible;
    row.querySelector(".col").value = toHex(r.base_color);
    row.querySelector(".rg").value = r.roughness;
    row.querySelector(".mt").value = r.metallic;
    row.querySelectorAll("output")[0].textContent = r.roughness.toFixed(2);
    row.querySelectorAll("output")[1].textContent = r.metallic.toFixed(2);
  });
  $(".f-screen").value = R.screen.content;
  $(".f-screen-color").value = toHex(R.screen.color);
  const bg = R.scene.background;
  $(".f-bgtype").value = bg.type;
  $(".f-bg0").value = bg.colors[0];
  $(".f-bg1").value = bg.colors[1];
  $(".f-bg1").hidden = bg.type === "solid";
  $(".f-env").value = R.scene.environment;
  $(".f-exposure").value = R.scene.exposure;
  $(".f-exposure").nextElementSibling.textContent = (+R.scene.exposure).toFixed(2);
  $(".f-ground").checked = R.scene.ground;
  $$(".f-pose").forEach((el) => { el.value = R.pose[el.dataset.k]; el.nextElementSibling.textContent = Math.round(R.pose[el.dataset.k]) + "°"; });
  $$("[data-r]").forEach((el) => (el.value = R.randomize[el.dataset.r][+el.dataset.i]));
  $$("[data-j]").forEach((el) => (el.value = R.randomize[el.dataset.j]));
  $$("[data-sc]").forEach((el) => (el.checked = R.randomize.screen_contents.includes(el.dataset.sc)));
  $(".f-rbg").checked = R.randomize.background;
  checkRanges();
}

function checkRanges() {
  $$(".rng").forEach((row) => {
    const [a, b] = Array.from(row.querySelectorAll("input"), (el) => parseFloat(el.value));
    row.classList.toggle("bad", !(a <= b));
  });
}

function select(i) {
  sel = sel === i ? -1 : i;
  syncControls();
  apply();
  const row = $(`.sdg-part[data-i="${sel}"]`);
  if (row) row.scrollIntoView({ block: "nearest" });
}

// ---- controls -> recipe
element.addEventListener("input", (e) => {
  const t = e.target;
  const row = t.closest(".sdg-part");
  if (row) {
    const r = R.parts[D.parts[+row.dataset.i].name];
    if (t.classList.contains("vis")) r.visible = t.checked;
    else if (t.classList.contains("col")) r.base_color = fromHex(t.value);
    else if (t.classList.contains("rg")) r.roughness = +t.value;
    else if (t.classList.contains("mt")) r.metallic = +t.value;
    syncControls(); apply(); return;
  }
  if (t.classList.contains("f-name")) R.name = t.value;
  else if (t.classList.contains("f-seed")) R.seed = Math.trunc(+t.value) || 0;
  else if (t.classList.contains("f-screen")) R.screen.content = t.value;
  else if (t.classList.contains("f-screen-color")) { R.screen.color = fromHex(t.value); R.screen.content = "solid"; }
  else if (t.classList.contains("f-bgtype")) R.scene.background.type = t.value;
  else if (t.classList.contains("f-bg0")) R.scene.background.colors[0] = t.value;
  else if (t.classList.contains("f-bg1")) R.scene.background.colors[1] = t.value;
  else if (t.classList.contains("f-env")) R.scene.environment = t.value;
  else if (t.classList.contains("f-exposure")) R.scene.exposure = +t.value;
  else if (t.classList.contains("f-ground")) R.scene.ground = t.checked;
  else if (t.classList.contains("f-pose")) R.pose[t.dataset.k] = +t.value;
  else if (t.classList.contains("f-mask")) mask = t.checked;
  else if (t.classList.contains("f-rbg")) R.randomize.background = t.checked;
  else if (t.dataset.r) { if (t.value !== "") R.randomize[t.dataset.r][+t.dataset.i] = +t.value; }
  else if (t.dataset.j) { if (t.value !== "") R.randomize[t.dataset.j] = +t.value; }
  else if (t.dataset.sc) {
    const set = new Set(R.randomize.screen_contents);
    t.checked ? set.add(t.dataset.sc) : set.delete(t.dataset.sc);
    R.randomize.screen_contents = CONTENTS.filter((c) => set.has(c));
  } else return;
  if (t.classList.contains("f-seed") || t.dataset.r || t.dataset.j || t.dataset.sc || t.classList.contains("f-rbg") || t.classList.contains("f-name")) {
    checkRanges(); return;  // randomization settings and names do not change the picture
  }
  syncControls(); apply();
});

async function randomize(next) {
  if (next) R.seed += 1;
  const res = await server.randomize({ recipe: R, seed: R.seed });
  showErrors(res.errors);
  if (!res.recipe) return;
  R = res.recipe;
  syncControls(); apply();
}

async function saveYaml() {
  const res = await server.dump_yaml(R);
  showErrors(res.errors);
  if (res.yaml == null) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([res.yaml], { type: "text/yaml" }));
  a.download = (R.name || "recipe") + ".yaml";
  a.click();
  URL.revokeObjectURL(a.href);
}

async function loadYaml(file) {
  const res = await server.load_yaml({ text: await file.text() });
  showErrors(res.errors);
  if (!res.recipe) return;
  R = res.recipe; texCache = {};
  syncControls(); apply();
}

element.addEventListener("change", (e) => {
  if (e.target.classList.contains("f-file") && e.target.files[0]) { loadYaml(e.target.files[0]); e.target.value = ""; }
});
element.addEventListener("click", (e) => {
  const nm = e.target.closest(".sdg-part .nm");
  if (nm) return select(+nm.parentElement.dataset.i);
  const act = e.target.closest("button")?.dataset.act;
  if (act === "load") $(".f-file").click();
  else if (act === "save") saveYaml();
  else if (act === "reset") { R = JSON.parse(JSON.stringify(D.recipe)); sel = -1; showErrors([]); syncControls(); apply(); }
  else if (act === "rand") randomize(false);
  else if (act === "rand-next") randomize(true);
  else if (act === "view-front") viewer.cameraOrbit = "90deg 80deg auto";
  else if (act === "view-iso") viewer.cameraOrbit = "45deg 65deg auto";
});

// ---- startup
function onModelLoad() {
  const all = viewer.model.materials;
  mats = D.parts.map((_, i) => all.find((m) => m.name === "p" + i) || null);
  apply();
}

async function start() {
  const view = $(".sdg-view");
  view.textContent = "loading...";
  try {
    D = await server.init("TV_a01");
    await import(D.mv);
  } catch (err) {
    view.textContent = "failed to start: " + err;
    return;
  }
  R = JSON.parse(JSON.stringify(D.recipe));
  buildStatic();
  syncControls();

  viewer = document.createElement("model-viewer");
  viewer.setAttribute("camera-controls", "");
  viewer.setAttribute("camera-orbit", "90deg 80deg auto");
  viewer.setAttribute("shadow-softness", "0.8");
  viewer.setAttribute("interaction-prompt", "none");
  viewer.addEventListener("load", onModelLoad);
  let down = null;
  viewer.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
  viewer.addEventListener("pointerup", (e) => {  // click (not drag) selects the part under the cursor
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const m = viewer.materialFromPoint(e.clientX, e.clientY);
    const i = m ? mats.indexOf(m) : -1;
    if (i >= 0 && R.parts[D.parts[i].name].visible) select(i);  // hidden parts (alpha 0) are not pickable
    else if (sel >= 0) select(sel);
  });
  view.replaceChildren(viewer);
  viewer.setAttribute("src", D.src);
}

start();
