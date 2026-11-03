// js_on_load for the gr.HTML viewer in app.py. props.value is a JSON string:
// {asset, src, mv, parts: [{name, verts, tris, color}]}. Opacity is applied to the loaded model's materials
// in the browser, so dragging a slider never rebuilds the GLB or resets the camera.
const st = { asset: null, op: [], vis: [], sel: -1, viewer: null, mats: [], base: [], data: null };
let mvLoad = null;

function apply() {
  st.mats.forEach((m, i) => {
    if (!m) return;
    const a = st.vis[i] ? st.op[i] : 0;
    m.setAlphaMode(a < 1 ? "BLEND" : "OPAQUE");
    m.pbrMetallicRoughness.setBaseColorFactor([...st.base[i].slice(0, 3), a]);
    m.setEmissiveFactor(st.sel === i ? [0.35, 0.3, 0] : [0, 0, 0]);
  });
  element.querySelectorAll(".sv-row").forEach((row) => {
    const i = +row.dataset.i;
    row.classList.toggle("sel", i === st.sel);
    row.querySelector("input.al").value = Math.round(st.op[i] * 100);
    row.querySelector("input.vis").checked = st.vis[i];
    row.querySelector("output").textContent = Math.round(st.op[i] * 100) + "%";
  });
}

function select(i) {
  st.sel = st.sel === i ? -1 : i;
  apply();
  const row = element.querySelector(`.sv-row[data-i="${st.sel}"]`);
  if (row) row.scrollIntoView({ block: "nearest" });
}

function buildList(parts) {
  const list = element.querySelector(".sv-list");
  list.replaceChildren(...parts.map((p, i) => {
    const row = document.createElement("div");
    row.className = "sv-row";
    row.dataset.i = i;
    row.innerHTML = `<input type="checkbox" class="vis" title="顯示 / 隱藏">
      <span class="sw"></span><span class="nm"></span>
      <input type="range" class="al" min="0" max="100"><output></output>`;
    row.querySelector(".sw").style.background = p.color;
    const nm = row.querySelector(".nm");
    nm.textContent = p.name;
    nm.title = `${p.name}\n頂點 ${p.verts.toLocaleString()}，三角形 ${p.tris.toLocaleString()}`;
    return row;
  }));
}

function onModelLoad() {
  const materials = st.viewer.model.materials;
  st.mats = st.data.parts.map((_, i) => materials.find((m) => m.name === "p" + i) || null);
  st.base = st.mats.map((m, i) => (m ? m.pbrMetallicRoughness.baseColorFactor : [0.8, 0.8, 0.8, 1]));
  apply();
}

function createViewer() {
  const v = document.createElement("model-viewer");
  v.setAttribute("camera-controls", "");
  v.setAttribute("environment-image", "neutral");
  v.setAttribute("shadow-intensity", "0");
  v.setAttribute("interaction-prompt", "none");
  v.addEventListener("load", onModelLoad);
  let down = null;
  v.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
  v.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const m = v.materialFromPoint(e.clientX, e.clientY);
    const i = m ? st.mats.indexOf(m) : -1;
    if (i >= 0) select(i);
    else if (st.sel >= 0) select(st.sel);  // click on background: deselect
  });
  return v;
}

async function render() {
  if (!props.value) return;
  const d = JSON.parse(props.value);
  const view = element.querySelector(".sv-view");
  try {
    await (mvLoad ??= import(d.mv));
  } catch (e) {
    mvLoad = null;
    view.textContent = "model-viewer 載入失敗：" + e;
    return;
  }
  st.data = d;
  st.viewer ??= createViewer();
  view.replaceChildren(st.viewer);
  if (st.asset !== d.asset) {  // new asset: reset; same asset (e.g. new decimation level): keep opacities
    st.asset = d.asset;
    st.op = d.parts.map(() => 1);
    st.vis = d.parts.map(() => true);
    st.sel = -1;
  }
  buildList(d.parts);
  if (st.viewer.getAttribute("src") === d.src && st.viewer.loaded) onModelLoad();
  else st.viewer.setAttribute("src", d.src);
}

element.addEventListener("input", (e) => {
  const row = e.target.closest(".sv-row");
  if (!row) return;
  const i = +row.dataset.i;
  if (e.target.classList.contains("al")) st.op[i] = e.target.value / 100;
  else if (e.target.classList.contains("vis")) st.vis[i] = e.target.checked;
  apply();
});
element.addEventListener("click", (e) => {
  const nm = e.target.closest(".nm");
  if (nm) return select(+nm.parentElement.dataset.i);
  const act = e.target.closest("button")?.dataset.act;
  if (act === "reset") {
    st.op = st.op.map(() => 1);
    st.vis = st.vis.map(() => true);
    apply();
  } else if (act === "solo" && st.sel >= 0) {
    st.op = st.op.map((_, i) => (i === st.sel ? 1 : 0.15));
    st.vis = st.vis.map(() => true);
    apply();
  }
});

watch("value", render);
render();
