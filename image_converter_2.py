import streamlit as st
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
import base64
from scipy.ndimage import gaussian_filter

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="3D Model Generator",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Minimal CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp {
        background: #0d0d0d;
        color: #f0f0f0;
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }

    /* Title */
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .app-sub {
        font-size: 0.95rem;
        color: #888;
        margin-bottom: 2rem;
    }

    /* Upload zone */
    .stFileUploader > section {
        background: #1a1a1a !important;
        border: 2px dashed #333 !important;
        border-radius: 12px !important;
        transition: border-color 0.2s ease !important;
    }
    .stFileUploader > section:hover {
        border-color: #555 !important;
    }

    /* Generate button */
    .stButton > button {
        width: 100%;
        background: #ffffff !important;
        color: #0d0d0d !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.5rem !important;
        transition: opacity 0.2s ease !important;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    /* Download button */
    .stDownloadButton > button {
        width: 100%;
        background: #1a1a1a !important;
        color: #f0f0f0 !important;
        font-weight: 500 !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.2rem !important;
        transition: border-color 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        border-color: #666 !important;
    }

    /* Stat chips */
    .stat-row {
        display: flex;
        gap: 0.75rem;
        margin-top: 0.75rem;
        flex-wrap: wrap;
    }
    .stat-chip {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 0.3rem 0.7rem;
        font-size: 0.8rem;
        color: #aaa;
    }
    .stat-chip span {
        color: #fff;
        font-weight: 600;
    }

    /* Section labels */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 0.5rem;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: #ffffff !important;
        border-radius: 4px !important;
    }

    /* Divider */
    hr { border-color: #1e1e1e; }

    /* Image preview */
    .stImage img {
        border-radius: 8px;
    }

    /* Error/warning */
    .stAlert {
        background: #1a1a1a !important;
        border-color: #333 !important;
        color: #ccc !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── 3D Processing (unchanged math) ─────────────────────────────────────────

def preprocess_image_for_3d(image, enhancement_type="edge_enhanced"):
    """Convert image to grayscale depth map + colour array."""
    image_array = np.array(image)

    if len(image_array.shape) == 3:
        if image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        color = image_array
    else:
        gray = image_array
        color = np.stack([gray, gray, gray], axis=-1)

    if enhancement_type == "edge_enhanced":
        edges = cv2.Canny(gray, 50, 150)
        gray = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)
    elif enhancement_type == "smooth_terrain":
        gray = gaussian_filter(gray, sigma=1.5)
    elif enhancement_type == "sharp_details":
        gray = cv2.equalizeHist(gray)
    elif enhancement_type == "artistic":
        gray = np.power(gray / 255.0, 0.7) * 255
        gray = gray.astype(np.uint8)

    return gray, color


def generate_mesh_from_pointcloud(points, colors, mesh_quality="medium"):
    """Build Open3D point cloud or Poisson mesh."""
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
    """Full pipeline: image → 3D model + metadata."""
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
        noise = np.random.normal(0, height_scale * 0.02, z_coords.shape)
        z_coords += noise

    points      = np.column_stack((x_coords.flatten(), y_coords.flatten(), z_coords.flatten()))
    colors_flat = color.reshape(-1, 3) / 255.0

    model, model_type = generate_mesh_from_pointcloud(points, colors_flat, mesh_quality)

    stats = {
        "total_points":  len(points),
        "model_type":    model_type,
        "width":         width,
        "height":        height,
        "orig_w":        original_w,
        "orig_h":        original_h,
        "downsample":    downsample,
    }
    return model, points, colors_flat, stats


def generate_ply_content(points, colors):
    """Serialise point cloud to ASCII PLY string."""
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header"
    )
    rows = []
    for pt, col in zip(points, colors):
        r, g, b = int(col[0] * 255), int(col[1] * 255), int(col[2] * 255)
        rows.append(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {r} {g} {b}")
    return header + "\n" + "\n".join(rows)


# ─── Three.js Viewer ─────────────────────────────────────────────────────────

def build_viewer_html(ply_content: str, height: int = 620) -> str:
    ply_b64 = base64.b64encode(ply_content.encode()).decode()

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d0d0d; overflow:hidden; font-family:'Inter',sans-serif; }}

  #canvas-wrap {{
    width:100%; height:{height}px;
    position:relative;
    border-radius:10px;
    overflow:hidden;
    border:1px solid #222;
  }}

  /* Loading overlay */
  #loader {{
    position:absolute; inset:0;
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    background:#0d0d0d; z-index:10; gap:14px;
  }}
  .spin {{
    width:32px; height:32px;
    border:3px solid #222;
    border-top-color:#fff;
    border-radius:50%;
    animation:spin 0.8s linear infinite;
  }}
  @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
  #loader-text {{ color:#666; font-size:13px; }}

  /* HUD */
  #hud {{
    position:absolute; bottom:16px; left:50%;
    transform:translateX(-50%);
    display:none;
    gap:6px;
    background:rgba(0,0,0,0.7);
    backdrop-filter:blur(8px);
    border:1px solid #2a2a2a;
    border-radius:8px;
    padding:6px 10px;
    align-items:center;
  }}
  .hud-btn {{
    background:#1e1e1e;
    border:1px solid #333;
    color:#ccc;
    font-size:11px;
    font-weight:500;
    border-radius:5px;
    padding:4px 10px;
    cursor:pointer;
    transition:all 0.15s;
    white-space:nowrap;
  }}
  .hud-btn:hover {{ background:#2a2a2a; color:#fff; }}
  .hud-btn.on {{ background:#fff; color:#000; border-color:#fff; }}
  .hud-sep {{ width:1px; height:16px; background:#2a2a2a; margin:0 2px; }}

  /* Point count badge */
  #badge {{
    position:absolute; top:14px; right:14px;
    background:rgba(0,0,0,0.7);
    backdrop-filter:blur(8px);
    border:1px solid #2a2a2a;
    border-radius:6px;
    padding:5px 10px;
    font-size:11px;
    color:#666;
    display:none;
  }}
  #badge span {{ color:#fff; font-weight:600; }}

  /* Hint */
  #hint {{
    position:absolute; top:14px; left:14px;
    background:rgba(0,0,0,0.7);
    backdrop-filter:blur(8px);
    border:1px solid #2a2a2a;
    border-radius:6px;
    padding:5px 10px;
    font-size:11px;
    color:#555;
    display:none;
  }}
</style>
</head>
<body>
<div id="canvas-wrap">
  <div id="loader">
    <div class="spin"></div>
    <div id="loader-text">Building 3D model…</div>
  </div>

  <div id="hud">
    <button class="hud-btn on" id="btn-rotate" onclick="toggleRotate()">⟳ Rotate</button>
    <div class="hud-sep"></div>
    <button class="hud-btn" id="btn-size-up"   onclick="changeSize(1)">+ Size</button>
    <button class="hud-btn" id="btn-size-down" onclick="changeSize(-1)">− Size</button>
    <div class="hud-sep"></div>
    <button class="hud-btn" id="btn-reset" onclick="resetView()">⊙ Reset</button>
    <div class="hud-sep"></div>
    <button class="hud-btn" id="btn-fs" onclick="toggleFS()">⤢ Full</button>
  </div>

  <div id="badge">Points: <span id="pt-count">—</span></div>
  <div id="hint">Drag · Scroll · Right-drag pan</div>
</div>

<script>
const PLY_B64 = "{ply_b64}";

let scene, camera, renderer, cloud;
let autoRotate = true;
let ptSize = 2.5;
let isDragging = false, isRight = false;
let prevMouse = {{x:0, y:0}};
let origCamPos = null;

function decode64(b64) {{
  const bin = atob(b64);
  let out = '';
  for (let i = 0; i < bin.length; i++) out += bin[i];
  return out;
}}

function parsePLY(text) {{
  const lines = text.split('\\n');
  let inHeader = true, vertCount = 0, hasNormals = false;
  const pos = [], col = [];

  for (const raw of lines) {{
    const line = raw.trim();
    if (inHeader) {{
      if (line.startsWith('element vertex')) vertCount = parseInt(line.split(' ')[2]);
      if (line.includes('property float nx')) hasNormals = true;
      if (line === 'end_header') {{ inHeader = false; continue; }}
    }} else {{
      if (pos.length / 3 >= vertCount) break;
      const p = line.split(' ');
      if (p.length < 6) continue;
      pos.push(parseFloat(p[0]), parseFloat(p[1]), parseFloat(p[2]));
      const off = hasNormals ? 6 : 3;
      col.push(parseInt(p[off])/255, parseInt(p[off+1])/255, parseInt(p[off+2])/255);
    }}
  }}

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('color',    new THREE.Float32BufferAttribute(col, 3));
  return geo;
}}

function init() {{
  scene    = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0d0d);

  const wrap = document.getElementById('canvas-wrap');
  const W = wrap.offsetWidth, H = {height};

  camera   = new THREE.PerspectiveCamera(60, W / H, 0.1, 5000);
  renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  wrap.appendChild(renderer.domElement);

  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const dl = new THREE.DirectionalLight(0xffffff, 0.8);
  dl.position.set(100, 100, 100);
  scene.add(dl);

  // Parse + add model
  setTimeout(() => {{
    document.getElementById('loader-text').textContent = 'Parsing point cloud…';
    setTimeout(() => {{
      try {{
        const geo  = parsePLY(decode64(PLY_B64));
        const mat  = new THREE.PointsMaterial({{ vertexColors:true, size:ptSize, sizeAttenuation:true }});
        cloud = new THREE.Points(geo, mat);
        scene.add(cloud);

        // Centre + fit camera
        const box    = new THREE.Box3().setFromObject(cloud);
        const centre = box.getCenter(new THREE.Vector3());
        const size   = box.getSize(new THREE.Vector3());
        cloud.position.sub(centre);
        const dist = Math.max(size.x, size.y, size.z) * 1.6;
        camera.position.set(dist * 0.7, dist * 0.5, dist);
        camera.lookAt(0, 0, 0);
        origCamPos = camera.position.clone();

        // Show UI
        document.getElementById('loader').style.display  = 'none';
        document.getElementById('hud').style.display     = 'flex';
        document.getElementById('badge').style.display   = 'block';
        document.getElementById('hint').style.display    = 'block';
        document.getElementById('pt-count').textContent  =
          (geo.attributes.position.count).toLocaleString();

        animate();
      }} catch(e) {{
        document.getElementById('loader-text').textContent = 'Error loading model';
        console.error(e);
      }}
    }}, 80);
  }}, 80);

  setupMouse();
  setupResize();
}}

function animate() {{
  requestAnimationFrame(animate);
  if (autoRotate && !isDragging && cloud) {{
    cloud.rotation.y += 0.004;
  }}
  renderer.render(scene, camera);
}}

// ── Controls ────────────────────────────────────────────────────────────────

function toggleRotate() {{
  autoRotate = !autoRotate;
  const b = document.getElementById('btn-rotate');
  b.classList.toggle('on', autoRotate);
  b.textContent = autoRotate ? '⟳ Rotate' : '⟳ Rotate';
}}

function changeSize(dir) {{
  ptSize = Math.max(0.5, Math.min(10, ptSize + dir * 0.5));
  if (cloud) cloud.material.size = ptSize;
}}

function resetView() {{
  if (!cloud || !origCamPos) return;
  camera.position.copy(origCamPos);
  camera.lookAt(0, 0, 0);
  cloud.rotation.set(0, 0, 0);
}}

function toggleFS() {{
  const wrap = document.getElementById('canvas-wrap');
  if (!document.fullscreenElement) {{
    wrap.requestFullscreen().catch(() => {{}});
    document.getElementById('btn-fs').textContent = '⤡ Exit';
  }} else {{
    document.exitFullscreen();
    document.getElementById('btn-fs').textContent = '⤢ Full';
  }}
}}

function setupMouse() {{
  const cv = renderer.domElement;
  cv.addEventListener('mousedown', e => {{
    isDragging = true; isRight = e.button === 2;
    prevMouse = {{x: e.clientX, y: e.clientY}};
    cv.style.cursor = 'grabbing';
    e.preventDefault();
  }});
  cv.addEventListener('mouseup',    () => {{ isDragging = false; cv.style.cursor = 'grab'; }});
  cv.addEventListener('mouseleave', () => {{ isDragging = false; cv.style.cursor = 'default'; }});
  cv.addEventListener('mousemove', e => {{
    if (!isDragging || !cloud) return;
    const dx = e.clientX - prevMouse.x, dy = e.clientY - prevMouse.y;
    if (isRight) {{
      camera.position.x -= dx * 0.08;
      camera.position.y += dy * 0.08;
    }} else {{
      cloud.rotation.y += dx * 0.008;
      cloud.rotation.x += dy * 0.008;
    }}
    prevMouse = {{x: e.clientX, y: e.clientY}};
    e.preventDefault();
  }});
  cv.addEventListener('wheel', e => {{
    e.preventDefault();
    camera.position.multiplyScalar(e.deltaY > 0 ? 1.08 : 0.92);
    const d = camera.position.length();
    if (d < 5)    camera.position.normalize().multiplyScalar(5);
    if (d > 2000) camera.position.normalize().multiplyScalar(2000);
  }}, {{passive:false}});
  cv.addEventListener('contextmenu', e => e.preventDefault());

  // Touch
  let lastPinch = 0;
  cv.addEventListener('touchstart', e => {{
    e.preventDefault();
    if (e.touches.length === 1) prevMouse = {{x: e.touches[0].clientX, y: e.touches[0].clientY}};
    if (e.touches.length === 2) {{
      const t = e.touches;
      lastPinch = Math.hypot(t[1].clientX - t[0].clientX, t[1].clientY - t[0].clientY);
    }}
  }}, {{passive:false}});
  cv.addEventListener('touchmove', e => {{
    e.preventDefault();
    if (e.touches.length === 1 && cloud) {{
      const dx = e.touches[0].clientX - prevMouse.x, dy = e.touches[0].clientY - prevMouse.y;
      cloud.rotation.y += dx * 0.01;
      cloud.rotation.x += dy * 0.01;
      prevMouse = {{x: e.touches[0].clientX, y: e.touches[0].clientY}};
    }}
    if (e.touches.length === 2) {{
      const t = e.touches;
      const d = Math.hypot(t[1].clientX - t[0].clientX, t[1].clientY - t[0].clientY);
      if (lastPinch) camera.position.multiplyScalar(lastPinch / d);
      lastPinch = d;
    }}
  }}, {{passive:false}});
}}

function setupResize() {{
  window.addEventListener('resize', () => {{
    const wrap = document.getElementById('canvas-wrap');
    const W = wrap.offsetWidth;
    const H = document.fullscreenElement ? window.innerHeight : {height};
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H);
  }});
  document.addEventListener('fullscreenchange', () => {{
    setTimeout(() => {{
      const wrap = document.getElementById('canvas-wrap');
      const W = wrap.offsetWidth;
      const H = document.fullscreenElement ? window.innerHeight : {height};
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
      renderer.setSize(W, H);
    }}, 100);
  }});
}}

init();
</script>
</body>
</html>"""


# ─── App UI ──────────────────────────────────────────────────────────────────

def main():
    st.markdown('<div class="app-title">3D Model Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Upload an image — get a 3D model.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.6], gap="large")

    # ── Left: upload + generate ──────────────────────────────────────────────
    with left:
        uploaded = st.file_uploader(
            "Drop an image here",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, use_column_width=True)

            w, h   = image.size
            mb     = len(uploaded.getvalue()) / (1024 * 1024)
            mp     = (w * h) / 1_000_000

            st.markdown(
                f'<div class="stat-row">'
                f'<div class="stat-chip"><span>{w}×{h}</span></div>'
                f'<div class="stat-chip"><span>{mp:.1f}</span> MP</div>'
                f'<div class="stat-chip"><span>{mb:.1f}</span> MB</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Generate 3D Model", type="primary"):
                with st.spinner(""):
                    prog = st.progress(0, text="Preprocessing image…")
                    prog.progress(15, text="Preprocessing image…")

                    model, points, colors_arr, stats = create_3d_model(
                        image,
                        height_scale=80,
                        enhancement_type="edge_enhanced",
                        mesh_quality="medium",
                        density_factor="medium",
                    )

                    prog.progress(70, text="Building point cloud…")

                    ply = generate_ply_content(points, colors_arr)

                    prog.progress(95, text="Rendering viewer…")

                    st.session_state.ply      = ply
                    st.session_state.points   = points
                    st.session_state.colors   = colors_arr
                    st.session_state.stats    = stats
                    st.session_state.filename = uploaded.name

                    prog.progress(100, text="Done!")
                    prog.empty()

            # Download once model exists
            if "ply" in st.session_state:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Download</div>', unsafe_allow_html=True)
                base = st.session_state.filename.rsplit(".", 1)[0]
                st.download_button(
                    "⬇  Download PLY",
                    data=st.session_state.ply.encode("utf-8"),
                    file_name=f"{base}_3d.ply",
                    mime="application/octet-stream",
                    use_container_width=True,
                )

        else:
            # Empty state hint
            st.markdown(
                """
                <div style="color:#444; font-size:0.85rem; text-align:center;
                            padding:2rem 0; line-height:1.8;">
                    PNG · JPG · BMP · TIFF · WebP<br>
                    <span style="font-size:0.75rem;">up to 200 MB</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Right: viewer ────────────────────────────────────────────────────────
    with right:
        if "ply" in st.session_state:
            s = st.session_state.stats

            # Mini stats bar
            st.markdown(
                f'<div class="stat-row" style="margin-bottom:0.75rem;">'
                f'<div class="stat-chip"><span>{s["total_points"]:,}</span> pts</div>'
                f'<div class="stat-chip"><span>{s["model_type"].title()}</span></div>'
                f'<div class="stat-chip"><span>{s["width"]}×{s["height"]}</span> sampled</div>'
                f'<div class="stat-chip">↓ <span>{s["downsample"]}×</span> downsample</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            html = build_viewer_html(st.session_state.ply, height=600)
            st.components.v1.html(html, height=608)

        else:
            # Placeholder
            st.markdown(
                """
                <div style="
                    height:600px; border:1px solid #1e1e1e; border-radius:10px;
                    display:flex; flex-direction:column;
                    align-items:center; justify-content:center;
                    color:#2a2a2a; gap:0.5rem;
                ">
                    <div style="font-size:2.5rem;">🧊</div>
                    <div style="font-size:0.85rem;">3D viewer will appear here</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
