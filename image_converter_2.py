import streamlit as st
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
import base64
from scipy.ndimage import gaussian_filter

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="3D Model Generator",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, .stApp { background:#0e0e0e; color:#f0f0f0; font-family:'Inter',sans-serif; }
    #MainMenu, footer, header { visibility:hidden; }

    .block-container { padding:2.5rem 3rem 3rem; max-width:1280px; }

    /* ── Header ── */
    .app-header {
        display:flex; align-items:baseline; gap:0.8rem;
        border-bottom:1px solid #1e1e1e;
        padding-bottom:1.2rem;
        margin-bottom:2rem;
    }
    .app-title  { font-size:1.5rem; font-weight:700; color:#fff; letter-spacing:-0.3px; margin:0; }
    .app-sub    { font-size:0.85rem; color:#555; margin:0; }

    /* ── Section labels ── */
    .panel-title {
        font-size:0.7rem; font-weight:600;
        letter-spacing:0.1em; text-transform:uppercase;
        color:#444; margin-bottom:0.75rem;
    }

    /* ── Upload zone ── */
    .stFileUploader > section {
        background:#141414 !important;
        border:1.5px dashed #2a2a2a !important;
        border-radius:10px !important;
        transition:border-color 0.2s !important;
        padding:1.2rem !important;
    }
    .stFileUploader > section:hover { border-color:#444 !important; }
    .stFileUploader label { display:none !important; }

    /* ── Image preview ── */
    .stImage img { border-radius:8px; width:100%; }

    /* ── Stat chips ── */
    .chip-row { display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.75rem; }
    .chip {
        background:#1a1a1a; border:1px solid #252525;
        border-radius:5px; padding:0.28rem 0.65rem;
        font-size:0.75rem; color:#666;
        display:flex; align-items:center; gap:0.3rem;
    }
    .chip b { color:#ccc; font-weight:600; }

    /* ── Divider ── */
    .divider { border:none; border-top:1px solid #1e1e1e; margin:1.1rem 0; }

    /* ── Primary button (Generate) ── */
    div[data-testid="stButton"] > button[kind="primary"] {
        width:100%;
        background:#fff !important;
        color:#0e0e0e !important;
        font-weight:600 !important;
        font-size:0.9rem !important;
        border:none !important;
        border-radius:8px !important;
        padding:0.7rem 1.5rem !important;
        letter-spacing:0.2px !important;
        transition:opacity 0.15s !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover { opacity:0.82 !important; }

    /* ── Secondary button ── */
    div[data-testid="stButton"] > button[kind="secondary"] {
        width:100%;
        background:transparent !important;
        color:#888 !important;
        font-weight:500 !important;
        font-size:0.85rem !important;
        border:1px solid #2a2a2a !important;
        border-radius:8px !important;
        padding:0.6rem 1.5rem !important;
        transition:all 0.15s !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        border-color:#444 !important; color:#ccc !important;
    }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        width:100%;
        background:#161616 !important;
        color:#aaa !important;
        font-weight:500 !important;
        font-size:0.85rem !important;
        border:1px solid #252525 !important;
        border-radius:8px !important;
        padding:0.58rem 1.2rem !important;
        transition:all 0.15s !important;
        text-align:left !important;
    }
    .stDownloadButton > button:hover {
        border-color:#444 !important; color:#fff !important;
        background:#1e1e1e !important;
    }

    /* ── Progress ── */
    .stProgress > div { background:#1e1e1e !important; border-radius:3px !important; }
    .stProgress > div > div { background:#fff !important; border-radius:3px !important; }

    /* ── Viewer placeholder ── */
    .viewer-placeholder {
        height:640px; border:1px solid #1a1a1a; border-radius:12px;
        display:flex; flex-direction:column;
        align-items:center; justify-content:center;
        gap:0.6rem; color:#252525;
    }
    .viewer-placeholder p { font-size:0.82rem; margin:0; }
</style>
""", unsafe_allow_html=True)


# ─── 3D Processing ────────────────────────────────────────────────────────────

def preprocess_image_for_3d(image, enhancement_type="edge_enhanced"):
    image_array = np.array(image)
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        gray  = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        color = image_array
    else:
        gray  = image_array
        color = np.stack([gray, gray, gray], axis=-1)

    if enhancement_type == "edge_enhanced":
        edges = cv2.Canny(gray, 50, 150)
        gray  = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)
    elif enhancement_type == "smooth_terrain":
        gray  = gaussian_filter(gray, sigma=1.5)
    elif enhancement_type == "sharp_details":
        gray  = cv2.equalizeHist(gray)
    elif enhancement_type == "artistic":
        gray  = (np.power(gray / 255.0, 0.7) * 255).astype(np.uint8)

    return gray, color


def generate_mesh_from_pointcloud(points, colors, mesh_quality="medium"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if mesh_quality == "point_cloud":
        return pcd, "point_cloud"

    try:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
        )
        depth_map = {"low": 6, "medium": 8, "high": 10}
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth_map.get(mesh_quality, 8)
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        return mesh, "mesh"
    except Exception:
        return pcd, "point_cloud"


def create_3d_model(image, height_scale=80, enhancement_type="edge_enhanced",
                    mesh_quality="medium", density_factor="medium"):
    gray, color = preprocess_image_for_3d(image, enhancement_type)

    original_h, original_w = gray.shape
    downsample_map = {
        "ultra_high": 1,
        "high":       max(1, max(original_w, original_h) // 800),
        "medium":     max(1, max(original_w, original_h) // 500),
        "low":        max(1, max(original_w, original_h) // 300),
        "preview":    max(1, max(original_w, original_h) // 150),
    }
    downsample = downsample_map.get(density_factor, 2)

    if downsample > 1:
        gray  = gray[::downsample, ::downsample]
        color = color[::downsample, ::downsample]

    height, width = gray.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    z_coords = gray.astype(np.float32) / 255.0 * height_scale

    if enhancement_type == "artistic":
        z_coords += np.random.normal(0, height_scale * 0.02, z_coords.shape)

    points      = np.column_stack((x_coords.flatten(), y_coords.flatten(), z_coords.flatten()))
    colors_flat = color.reshape(-1, 3) / 255.0

    model, model_type = generate_mesh_from_pointcloud(points, colors_flat, mesh_quality)

    stats = {
        "total_points": len(points),
        "model_type":   model_type,
        "width":        width,
        "height":       height,
        "orig_w":       original_w,
        "orig_h":       original_h,
        "downsample":   downsample,
    }
    return model, points, colors_flat, stats


def generate_ply_content(points, colors):
    """Serialise point cloud to ASCII PLY."""
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header"
    )
    rows = [
        f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} "
        f"{int(c[0]*255)} {int(c[1]*255)} {int(c[2]*255)}"
        for pt, c in zip(points, colors)
    ]
    return header + "\n" + "\n".join(rows)


def generate_mesh_ply(model) -> str:
    """Serialise an Open3D TriangleMesh to ASCII PLY."""
    verts = np.asarray(model.vertices)
    tris  = np.asarray(model.triangles)
    vc    = np.asarray(model.vertex_colors) if model.has_vertex_colors() else None

    lines = [
        "ply", "format ascii 1.0",
        f"element vertex {len(verts)}",
        "property float x", "property float y", "property float z",
    ]
    if vc is not None:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines += [f"element face {len(tris)}", "property list uchar int vertex_indices", "end_header"]

    for i, v in enumerate(verts):
        if vc is not None:
            r, g, b = int(vc[i, 0]*255), int(vc[i, 1]*255), int(vc[i, 2]*255)
            lines.append(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {r} {g} {b}")
        else:
            lines.append(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
    for t in tris:
        lines.append(f"3 {t[0]} {t[1]} {t[2]}")

    return "\n".join(lines)


# ─── Three.js Viewer ──────────────────────────────────────────────────────────

def build_viewer_html(ply_content: str, mesh_ply: str, height: int = 640) -> str:
    pts_b64  = base64.b64encode(ply_content.encode()).decode()
    mesh_b64 = base64.b64encode(mesh_ply.encode()).decode() if mesh_ply else ""
    has_mesh = "true" if mesh_ply else "false"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
  *{{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0e0e0e; overflow:hidden; font-family:'Inter',system-ui,sans-serif; }}

  #wrap {{
    width:100%; height:{height}px; position:relative;
    border-radius:12px; overflow:hidden;
    background:#0e0e0e; border:1px solid #1e1e1e;
  }}

  /* Loader */
  #loader {{
    position:absolute; inset:0; z-index:20;
    display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:16px;
    background:#0e0e0e;
  }}
  .spin {{
    width:26px; height:26px;
    border:2px solid #222; border-top-color:#fff;
    border-radius:50%; animation:spin 0.75s linear infinite;
  }}
  @keyframes spin{{ to{{ transform:rotate(360deg); }} }}
  #loader-msg {{ font-size:11.5px; color:#444; letter-spacing:0.06em; }}
  #loader-track {{
    width:140px; height:2px; background:#1a1a1a;
    border-radius:1px; overflow:hidden;
  }}
  #loader-fill {{
    height:100%; width:0%; background:#fff;
    border-radius:1px; transition:width 0.25s ease;
  }}

  /* Info pill — top left */
  #info-pill {{
    position:absolute; top:14px; left:14px; z-index:10;
    display:none; flex-direction:column; gap:5px;
  }}
  .pill {{
    background:rgba(8,8,8,0.88); backdrop-filter:blur(14px);
    border:1px solid #1e1e1e; border-radius:6px;
    padding:5px 10px; font-size:10.5px; color:#444; line-height:1.5;
    white-space:nowrap;
  }}
  .pill b {{ color:#999; font-weight:600; }}

  /* Point-size stepper — top right */
  #pt-stepper {{
    position:absolute; top:14px; right:14px; z-index:10;
    display:none; align-items:center;
    background:rgba(8,8,8,0.88); backdrop-filter:blur(14px);
    border:1px solid #1e1e1e; border-radius:7px; overflow:hidden;
  }}
  .pt-btn {{
    background:transparent; border:none; color:#555;
    font-size:15px; width:30px; height:28px;
    cursor:pointer; transition:all 0.12s; line-height:1;
    display:flex; align-items:center; justify-content:center;
  }}
  .pt-btn:hover {{ background:#1e1e1e; color:#fff; }}
  #pt-val {{
    font-size:10.5px; color:#555; padding:0 8px;
    border-left:1px solid #1e1e1e; border-right:1px solid #1e1e1e;
    line-height:1; user-select:none; white-space:nowrap;
    min-width:36px; text-align:center;
  }}

  /* Bottom toolbar */
  #toolbar {{
    position:absolute; bottom:16px; left:50%;
    transform:translateX(-50%);
    z-index:10; display:none;
    background:rgba(6,6,6,0.92); backdrop-filter:blur(18px);
    border:1px solid #222; border-radius:10px;
    padding:6px 8px; align-items:center; gap:3px;
    box-shadow:0 4px 28px rgba(0,0,0,0.6);
  }}

  .tb-group {{ display:flex; gap:3px; align-items:center; }}
  .tb-sep {{ width:1px; height:16px; background:#1e1e1e; margin:0 5px; flex-shrink:0; }}

  /* Regular toolbar button */
  .tb-btn {{
    background:transparent; border:1px solid transparent;
    color:#555; font-size:11px; font-weight:500;
    border-radius:6px; padding:5px 12px;
    cursor:pointer; transition:all 0.13s;
    white-space:nowrap; line-height:1;
    display:flex; align-items:center; gap:5px;
  }}
  .tb-btn:hover {{ background:#1c1c1c; color:#ccc; border-color:#2a2a2a; }}
  .tb-btn.active {{
    background:#ffffff; color:#0e0e0e;
    border-color:#ffffff; font-weight:600;
  }}

  /* Icon-only button */
  .tb-icon {{
    background:transparent; border:1px solid transparent;
    color:#555; font-size:13px;
    border-radius:6px; padding:5px 7px;
    cursor:pointer; transition:all 0.13s; line-height:1;
  }}
  .tb-icon:hover {{ background:#1c1c1c; color:#ccc; border-color:#2a2a2a; }}
  .tb-icon.active {{ background:#fff; color:#0e0e0e; border-color:#fff; }}
</style>
</head>
<body>
<div id="wrap">

  <div id="loader">
    <div class="spin"></div>
    <div id="loader-msg">Initialising…</div>
    <div id="loader-track"><div id="loader-fill"></div></div>
  </div>

  <div id="info-pill">
    <div class="pill" id="pill-mode">Mode: <b>Points</b></div>
    <div class="pill" id="pill-pts">— pts</div>
  </div>

  <div id="pt-stepper">
    <button class="pt-btn" onclick="chSz(-1)">−</button>
    <span id="pt-val">2.5</span>
    <button class="pt-btn" onclick="chSz(1)">+</button>
  </div>

  <div id="toolbar">
    <div class="tb-group">
      <button class="tb-btn active" id="btn-pts"  onclick="setMode('points')">⬤ Points</button>
      <button class="tb-btn"        id="btn-mesh" onclick="setMode('mesh')">⬡ Mesh</button>
    </div>
    <div class="tb-sep"></div>
    <div class="tb-group">
      <button class="tb-btn active" id="btn-spin" onclick="toggleSpin()">↻ Spin</button>
    </div>
    <div class="tb-sep"></div>
    <div class="tb-group">
      <button class="tb-btn"  id="btn-reset" onclick="resetView()">⊙ Reset</button>
      <button class="tb-icon" id="btn-fs"    onclick="toggleFS()" title="Fullscreen">⤢</button>
    </div>
  </div>

</div>

<script>
const PTS_B64  = "{pts_b64}";
const MESH_B64 = "{mesh_b64}";
const HAS_MESH = {has_mesh};

let scene, camera, renderer;
let ptCloud = null, meshObj = null;
let viewMode   = 'points';
let autoSpin   = true;
let ptSize     = 2.5;
let isDrag = false, isRight = false;
let prev = {{x:0, y:0}};
let origCam = null;

// ── Progress helpers ──────────────────────────────────────────────────────────
const setBar = p => document.getElementById('loader-fill').style.width = p + '%';
const setMsg = t => document.getElementById('loader-msg').textContent = t;

// ── Circle sprite (soft disc, not aliased square) ─────────────────────────────
function makeDisc(sz=64) {{
  const c = document.createElement('canvas');
  c.width = c.height = sz;
  const ctx = c.getContext('2d'), r = sz/2;
  const g = ctx.createRadialGradient(r,r,0, r,r,r);
  g.addColorStop(0,    'rgba(255,255,255,1.0)');
  g.addColorStop(0.40, 'rgba(255,255,255,0.98)');
  g.addColorStop(0.72, 'rgba(255,255,255,0.55)');
  g.addColorStop(0.90, 'rgba(255,255,255,0.10)');
  g.addColorStop(1.0,  'rgba(255,255,255,0.0)');
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(r,r,r,0,Math.PI*2); ctx.fill();
  return new THREE.CanvasTexture(c);
}}
const discTex = makeDisc(64);

// ── PLY parsers ───────────────────────────────────────────────────────────────
function parsePts(text) {{
  const lines = text.split('\\n');
  let hdr=true, nV=0, hasN=false;
  const pos=[], col=[];
  for (const raw of lines) {{
    const l = raw.trim();
    if (hdr) {{
      if (l.startsWith('element vertex')) nV = parseInt(l.split(' ')[2]);
      if (l.includes('property float nx')) hasN = true;
      if (l === 'end_header') {{ hdr=false; continue; }}
    }} else {{
      if (pos.length/3 >= nV) break;
      const p = l.split(' '); if (p.length<6) continue;
      pos.push(parseFloat(p[0]), parseFloat(p[1]), parseFloat(p[2]));
      const o = hasN ? 6 : 3;
      col.push(parseInt(p[o])/255, parseInt(p[o+1])/255, parseInt(p[o+2])/255);
    }}
  }}
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos,3));
  g.setAttribute('color',    new THREE.Float32BufferAttribute(col,3));
  return g;
}}

function parseMesh(text) {{
  const lines = text.split('\\n');
  let hdr=true, nV=0, nF=0, hasC=false, vDone=0;
  const pos=[], col=[], idx=[];
  for (const raw of lines) {{
    const l = raw.trim();
    if (hdr) {{
      if (l.startsWith('element vertex')) nV = parseInt(l.split(' ')[2]);
      if (l.startsWith('element face'))   nF = parseInt(l.split(' ')[2]);
      if (l.includes('property uchar red')) hasC = true;
      if (l === 'end_header') {{ hdr=false; continue; }}
    }} else {{
      if (vDone < nV) {{
        const p = l.split(' ');
        pos.push(parseFloat(p[0]), parseFloat(p[1]), parseFloat(p[2]));
        if (hasC && p.length>=6)
          col.push(parseInt(p[3])/255, parseInt(p[4])/255, parseInt(p[5])/255);
        else col.push(0.72, 0.72, 0.72);
        vDone++;
      }} else {{
        const p = l.split(' ');
        if (p.length>=4 && parseInt(p[0])===3)
          idx.push(parseInt(p[1]), parseInt(p[2]), parseInt(p[3]));
      }}
    }}
  }}
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos,3));
  g.setAttribute('color',    new THREE.Float32BufferAttribute(col,3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}}

// ── Init ──────────────────────────────────────────────────────────────────────
function init() {{
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e0e0e);

  const wrap = document.getElementById('wrap');
  const W = wrap.offsetWidth, H = {height};
  camera = new THREE.PerspectiveCamera(55, W/H, 0.1, 10000);

  renderer = new THREE.WebGLRenderer({{ antialias:true }});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W, H);
  wrap.appendChild(renderer.domElement);

  // Lights (for mesh mode)
  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.75);
  dl1.position.set(200, 200, 200); scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0x9aabff, 0.30);
  dl2.position.set(-120, -80, 80); scene.add(dl2);

  setMsg('Parsing point cloud…'); setBar(20);

  setTimeout(() => {{
    try {{
      // ── Points ──────────────────────────────────────────────────────────
      const pGeo = parsePts(atob(PTS_B64));
      setBar(48);

      const pMat = new THREE.PointsMaterial({{
        vertexColors:    true,
        size:            ptSize,
        sizeAttenuation: true,
        map:             discTex,
        alphaTest:       0.04,
        transparent:     true,
        depthWrite:      false,
      }});
      ptCloud = new THREE.Points(pGeo, pMat);

      // Compute bounding box for centering
      pGeo.computeBoundingBox();
      const bb  = pGeo.boundingBox;
      const cen = new THREE.Vector3();
      bb.getCenter(cen);
      const sz  = new THREE.Vector3();
      bb.getSize(sz);

      // Centre geometry in-place so position=(0,0,0) after this
      pGeo.translate(-cen.x, -cen.y, -cen.z);

      const dist = Math.max(sz.x, sz.y, sz.z) * 1.75;
      camera.position.set(dist*0.6, dist*0.45, dist);
      camera.lookAt(0,0,0);
      origCam = camera.position.clone();

      scene.add(ptCloud);
      document.getElementById('pill-pts').innerHTML =
        '<b>' + pGeo.attributes.position.count.toLocaleString() + '</b> points';

      // ── Mesh ────────────────────────────────────────────────────────────
      if (HAS_MESH && MESH_B64) {{
        setMsg('Building mesh…'); setBar(68);
        const mGeo = parseMesh(atob(MESH_B64));

        // Centre mesh geometry the same way
        mGeo.computeBoundingBox();
        const mc = new THREE.Vector3();
        mGeo.boundingBox.getCenter(mc);
        mGeo.translate(-mc.x, -mc.y, -mc.z);

        meshObj = new THREE.Mesh(mGeo, new THREE.MeshPhongMaterial({{
          vertexColors: true,
          shininess:    22,
          specular:     new THREE.Color(0x1a1a1a),
          side:         THREE.DoubleSide,
        }}));
        meshObj.visible = false;
        scene.add(meshObj);
      }}

      setBar(96);
      setTimeout(() => {{
        document.getElementById('loader').style.display    = 'none';
        document.getElementById('toolbar').style.display   = 'flex';
        document.getElementById('info-pill').style.display = 'flex';
        document.getElementById('pt-stepper').style.display= 'flex';

        if (!HAS_MESH) {{
          const b = document.getElementById('btn-mesh');
          b.style.opacity       = '0.3';
          b.style.pointerEvents = 'none';
          b.title = 'Regenerate with mesh quality to enable';
        }}
        animate();
      }}, 180);

    }} catch(e) {{
      setMsg('⚠ Error loading model'); console.error(e);
    }}
  }}, 100);

  setupMouse();
  setupResize();
}}

// ── Render loop ───────────────────────────────────────────────────────────────
function animate() {{
  requestAnimationFrame(animate);
  const obj = viewMode==='mesh' ? meshObj : ptCloud;
  if (autoSpin && !isDrag && obj) obj.rotation.y += 0.0032;
  renderer.render(scene, camera);
}}

// ── View mode switch ──────────────────────────────────────────────────────────
function setMode(m) {{
  if (m==='mesh' && !HAS_MESH) return;
  viewMode = m;
  const pts = m==='points';

  if (ptCloud) ptCloud.visible = pts;
  if (meshObj) {{
    meshObj.visible = !pts;
    // Sync rotation between objects
    if (pts && ptCloud) meshObj.rotation.copy(ptCloud.rotation);
    else if (ptCloud)    ptCloud.rotation.copy(meshObj.rotation);
  }}

  document.getElementById('btn-pts').classList.toggle('active', pts);
  document.getElementById('btn-mesh').classList.toggle('active', !pts);
  document.getElementById('pill-mode').innerHTML =
    'Mode: <b>' + (pts ? 'Points' : 'Mesh') + '</b>';

  // Stepper only useful in points mode
  const s = document.getElementById('pt-stepper');
  s.style.opacity       = pts ? '1' : '0.35';
  s.style.pointerEvents = pts ? 'auto' : 'none';
}}

// ── Button handlers ───────────────────────────────────────────────────────────
function toggleSpin() {{
  autoSpin = !autoSpin;
  document.getElementById('btn-spin').classList.toggle('active', autoSpin);
}}

function chSz(d) {{
  ptSize = Math.max(0.5, Math.min(12, ptSize + d*0.5));
  if (ptCloud) ptCloud.material.size = ptSize;
  document.getElementById('pt-val').textContent = ptSize.toFixed(1);
}}

function resetView() {{
  if (!origCam) return;
  camera.position.copy(origCam); camera.lookAt(0,0,0);
  const obj = viewMode==='mesh' ? meshObj : ptCloud;
  if (obj) obj.rotation.set(0,0,0);
  if (ptCloud && meshObj) {{
    ptCloud.rotation.set(0,0,0); meshObj.rotation.set(0,0,0);
  }}
}}

function toggleFS() {{
  const w = document.getElementById('wrap');
  if (!document.fullscreenElement) {{
    w.requestFullscreen().catch(()=>{{}});
    document.getElementById('btn-fs').textContent='⤡';
  }} else {{
    document.exitFullscreen();
    document.getElementById('btn-fs').textContent='⤢';
  }}
}}

// ── Mouse & touch ─────────────────────────────────────────────────────────────
function activeObj() {{ return viewMode==='mesh' ? meshObj : ptCloud; }}

function syncRot() {{
  if (!ptCloud || !meshObj) return;
  if (viewMode==='points') meshObj.rotation.copy(ptCloud.rotation);
  else ptCloud.rotation.copy(meshObj.rotation);
}}

function setupMouse() {{
  const cv = renderer.domElement;

  cv.addEventListener('mousedown', e => {{
    isDrag=true; isRight=e.button===2;
    prev={{x:e.clientX, y:e.clientY}};
    cv.style.cursor='grabbing'; e.preventDefault();
  }});
  cv.addEventListener('mouseup',    ()=>{{ isDrag=false; cv.style.cursor='grab'; }});
  cv.addEventListener('mouseleave', ()=>{{ isDrag=false; cv.style.cursor='default'; }});
  cv.addEventListener('mousemove', e => {{
    if (!isDrag) return;
    const dx=e.clientX-prev.x, dy=e.clientY-prev.y;
    const obj=activeObj();
    if (isRight) {{
      camera.position.x -= dx*0.07;
      camera.position.y += dy*0.07;
    }} else if (obj) {{
      obj.rotation.y += dx*0.007;
      obj.rotation.x += dy*0.007;
      syncRot();
    }}
    prev={{x:e.clientX, y:e.clientY}}; e.preventDefault();
  }});
  cv.addEventListener('wheel', e => {{
    e.preventDefault();
    camera.position.multiplyScalar(e.deltaY>0 ? 1.08 : 0.93);
    const d=camera.position.length();
    if (d<3)    camera.position.normalize().multiplyScalar(3);
    if (d>5000) camera.position.normalize().multiplyScalar(5000);
  }}, {{passive:false}});
  cv.addEventListener('contextmenu', e=>e.preventDefault());

  let lp=0;
  cv.addEventListener('touchstart', e=>{{
    e.preventDefault();
    if (e.touches.length===1) prev={{x:e.touches[0].clientX, y:e.touches[0].clientY}};
    if (e.touches.length===2) {{
      const t=e.touches;
      lp=Math.hypot(t[1].clientX-t[0].clientX, t[1].clientY-t[0].clientY);
    }}
  }}, {{passive:false}});
  cv.addEventListener('touchmove', e=>{{
    e.preventDefault();
    const obj=activeObj();
    if (e.touches.length===1 && obj) {{
      const dx=e.touches[0].clientX-prev.x, dy=e.touches[0].clientY-prev.y;
      obj.rotation.y+=dx*0.01; obj.rotation.x+=dy*0.01; syncRot();
      prev={{x:e.touches[0].clientX, y:e.touches[0].clientY}};
    }}
    if (e.touches.length===2) {{
      const t=e.touches;
      const d=Math.hypot(t[1].clientX-t[0].clientX, t[1].clientY-t[0].clientY);
      if (lp) camera.position.multiplyScalar(lp/d);
      lp=d;
    }}
  }}, {{passive:false}});
}}

function setupResize() {{
  const doResize = () => {{
    const w=document.getElementById('wrap');
    const W=w.offsetWidth, H=document.fullscreenElement?window.innerHeight:{height};
    camera.aspect=W/H; camera.updateProjectionMatrix();
    renderer.setSize(W,H);
  }};
  window.addEventListener('resize', doResize);
  document.addEventListener('fullscreenchange', ()=>setTimeout(doResize,100));
}}

init();
</script>
</body>
</html>"""


# ─── App ─────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div class="app-header">
        <span class="app-title">3D Model Generator</span>
        <span class="app-sub">Upload an image · Get an interactive 3D model</span>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([5, 8], gap="large")

    # ── Left ──────────────────────────────────────────────────────────────────
    with left:

        st.markdown('<div class="panel-title">Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "upload",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, use_column_width=True)

            w, h = image.size
            mb   = len(uploaded.getvalue()) / (1024 * 1024)
            mp   = (w * h) / 1_000_000

            st.markdown(
                f'<div class="chip-row">'
                f'<div class="chip"><b>{w}×{h}</b></div>'
                f'<div class="chip"><b>{mp:.1f}</b> MP</div>'
                f'<div class="chip"><b>{mb:.1f}</b> MB</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            if st.button("Generate 3D Model", type="primary", use_container_width=True):
                bar = st.progress(0, text="Preprocessing image…")
                bar.progress(10, text="Preprocessing image…")

                model, points, colors_arr, stats = create_3d_model(
                    image,
                    height_scale     = 80,
                    enhancement_type = "edge_enhanced",
                    mesh_quality     = "medium",
                    density_factor   = "medium",
                )
                bar.progress(58, text="Serialising point cloud…")
                pts_ply = generate_ply_content(points, colors_arr)

                mesh_ply = ""
                if stats["model_type"] == "mesh":
                    bar.progress(76, text="Serialising mesh…")
                    mesh_ply = generate_mesh_ply(model)

                bar.progress(96, text="Preparing viewer…")
                st.session_state.pts_ply  = pts_ply
                st.session_state.mesh_ply = mesh_ply
                st.session_state.stats    = stats
                st.session_state.filename = uploaded.name
                bar.progress(100, text="Done ✓")
                bar.empty()
                st.rerun()

            # Results section
            if "stats" in st.session_state:
                s = st.session_state.stats

                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<div class="panel-title">Model Info</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="chip-row">'
                    f'<div class="chip"><b>{s["total_points"]:,}</b> points</div>'
                    f'<div class="chip"><b>{s["model_type"].title()}</b></div>'
                    f'<div class="chip"><b>{s["orig_w"]}×{s["orig_h"]}</b> original</div>'
                    f'<div class="chip"><b>{s["width"]}×{s["height"]}</b> sampled</div>'
                    f'<div class="chip">↓ <b>{s["downsample"]}×</b></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<div class="panel-title">Export</div>', unsafe_allow_html=True)

                base = st.session_state.filename.rsplit(".", 1)[0]
                st.download_button(
                    "⬇  Point Cloud (.ply)",
                    data=st.session_state.pts_ply.encode("utf-8"),
                    file_name=f"{base}_pointcloud.ply",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
                if st.session_state.mesh_ply:
                    st.download_button(
                        "⬇  Mesh (.ply)",
                        data=st.session_state.mesh_ply.encode("utf-8"),
                        file_name=f"{base}_mesh.ply",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )

                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                if st.button("↺  Regenerate", type="secondary", use_container_width=True):
                    for k in ["pts_ply", "mesh_ply", "stats", "filename"]:
                        st.session_state.pop(k, None)
                    st.rerun()

        else:
            st.markdown(
                '<div style="color:#2a2a2a; font-size:0.8rem; text-align:center;'
                ' padding:1.5rem 0; line-height:2.2;">'
                'PNG · JPG · JPEG · BMP · TIFF · WebP<br>'
                '<span style="font-size:0.72rem; color:#222;">up to 200 MB</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── Right: viewer ─────────────────────────────────────────────────────────
    with right:
        if "pts_ply" in st.session_state:
            html = build_viewer_html(
                st.session_state.pts_ply,
                st.session_state.mesh_ply,
                height=640,
            )
            st.components.v1.html(html, height=648)
        else:
            st.markdown("""
            <div class="viewer-placeholder">
                <svg width="48" height="48" viewBox="0 0 48 48" fill="none"
                     xmlns="http://www.w3.org/2000/svg">
                  <path d="M24 4 L42 14 L42 34 L24 44 L6 34 L6 14 Z"
                        stroke="#fff" stroke-width="1.2" fill="none"/>
                  <path d="M24 4 L24 44 M6 14 L42 34 M42 14 L6 34"
                        stroke="#fff" stroke-width="0.6" stroke-dasharray="3 3" opacity="0.4"/>
                </svg>
                <p>3D viewer will appear here</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()