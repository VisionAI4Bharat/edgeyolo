/*
 * Copyright (C) 2026 swatah.ai. All rights reserved.
 *
 * This software is dual-licensed:
 * 1. GNU General Public License v3.0 (GPLv3)
 * 2. A proprietary license for commercial use.
 *
 * You may use this software under the terms of the GPLv3 if you are using it
 * for non-commercial purposes. For commercial usage, a separate commercial
 * license must be obtained from swatah.ai (info@swatah.ai).
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * Trademarks: All trademarks, service marks, and logos are the property of
 * their respective owners.
 */

#include "web_config.h"
#include "headless_app.h"
#include "../app_config.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ── HTML dashboard (served at GET /) ─────────────────────────────────────────

static const char kDashboard[] = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>deepSightAI Config</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111827;color:#e5e7eb;font:14px/1.6 system-ui,sans-serif;padding:16px;max-width:640px;margin:0 auto}
h1{color:#60a5fa;font-size:18px;margin-bottom:16px}
.card{background:#1f2937;border-radius:8px;padding:16px;margin-bottom:12px}
.card h2{color:#93c5fd;font-size:11px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;border-bottom:1px solid #374151;padding-bottom:6px}
.row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.row label{min-width:140px;color:#9ca3af;font-size:13px}
input,select{flex:1;background:#374151;border:1px solid #4b5563;color:#e5e7eb;border-radius:4px;padding:5px 8px;font-size:13px}
input[type=checkbox]{flex:none;width:18px;height:18px;accent-color:#2563eb}
.btn{background:#2563eb;color:#fff;border:none;border-radius:4px;padding:6px 14px;cursor:pointer;font-size:13px;margin-top:4px}
.btn:hover{background:#1d4ed8}
.btn.red{background:#dc2626}.btn.red:hover{background:#b91c1c}
.st{font-size:12px;color:#6b7280;margin-top:6px;min-height:16px}
#topbar{display:flex;justify-content:space-between;align-items:center;background:#1f2937;border-radius:6px;padding:8px 14px;margin-bottom:14px;font-size:13px}
#conn{color:#6b7280}
</style>
</head>
<body>
<div id="topbar"><b style="color:#60a5fa">deepSightAI</b><span id="conn">connecting&#8230;</span></div>

<div class="card"><h2>Model</h2>
<div class="row"><label>Backend</label><select id="backend"><option value="0">ONNX Runtime</option><option value="1">OpenVINO</option><option value="2">RKNN</option></select></div>
<div class="row"><label>Rockchip HW</label><input id="rockchip_hw" type="checkbox"></div>
<div class="row"><label>Model file</label><input id="model_file" type="text" placeholder="/path/to/model.rknn"></div>
<div class="row"><label>YAML file</label><input id="yaml_file" type="text" placeholder="/path/to/model.yaml"></div>
<div class="row"><label>Class labels (csv)</label><input id="class_labels" type="text" placeholder="person,car,bike"></div>
<button class="btn" onclick="dsai_applyModel()">Apply</button>
<div class="st" id="st-m"></div></div>

<div class="card"><h2>Video Source</h2>
<div class="row"><label>Source</label><select id="source" onchange="srcVis()"><option value="0">Camera (VI)</option><option value="1">Video file</option><option value="2">RTSP stream</option></select></div>
<div class="row" id="r-cam"><label>Camera device ID</label><input id="camera_device_id" type="number" min="0"></div>
<div class="row" id="r-rtsp"><label>RTSP URL</label><input id="rtsp_url" type="text" placeholder="rtsp://192.168.1.x:554/stream"></div>
<div class="row" id="r-vid"><label>Video file path</label><input id="video_file" type="text"></div>
<div class="row"><label>Gain</label><input id="gain" type="number" min="0" max="255"></div>
<div class="row"><label>Gamma</label><input id="gamma" type="number" min="72" max="500"></div>
<div class="row"><label>Brightness</label><input id="brightness" type="number" min="-64" max="64"></div>
<div class="row"><label>Resolution</label>
<select id="resolution_index"><option value="0">640x480</option><option value="1">1280x720</option><option value="2">1920x1080</option><option value="3">320x240</option><option value="4">416x416</option></select></div>
<div class="row"><label>FPS</label><select id="fps_index"><option value="0">15</option><option value="1">25</option><option value="2">30</option><option value="3">60</option><option value="4">90</option></select></div>
<div class="row"><label>IQ files dir</label><input id="iq_dir" type="text"></div>
<button class="btn" onclick="dsai_applySource()">Apply</button>
<div class="st" id="st-s"></div></div>

<div class="card"><h2>Detection</h2>
<div class="row"><label>Conf threshold</label><input id="conf_threshold" type="number" step="0.01" min="0" max="1"></div>
<div class="row"><label>NMS threshold</label><input id="nms_threshold" type="number" step="0.01" min="0" max="1"></div>
<button class="btn" onclick="applyDetection()">Apply</button>
<div class="st" id="st-d"></div></div>

<div class="card"><h2>Region of Interest</h2>
<div class="row"><label>Enabled</label><input id="roi_enabled" type="checkbox"></div>
<div class="row"><label>X</label><input id="roi_x" type="number" min="0" value="0"></div>
<div class="row"><label>Y</label><input id="roi_y" type="number" min="0" value="0"></div>
<div class="row"><label>Width</label><input id="roi_w" type="number" min="0" value="0"></div>
<div class="row"><label>Height</label><input id="roi_h" type="number" min="0" value="0"></div>
<button class="btn" onclick="dsai_applyRoi()">Apply</button>
<div class="st" id="st-r"></div></div>

<div class="card"><h2>System</h2>
<div class="row"><label>Web port</label><input id="web_port" type="number" min="1024" max="65535"></div>
<div class="row"><label>Debug logging</label><input id="debug_logging" type="checkbox"></div>
<button class="btn" onclick="applySystem()">Apply</button>
<div class="st" id="st-sy"></div></div>

<div class="card"><h2>Control</h2>
<p style="color:#9ca3af;font-size:13px;margin-bottom:10px">Restart relaunches the inference pipeline with all saved settings.</p>
<button class="btn red" onclick="doRestart()">Restart Application</button>
<div class="st" id="st-x"></div></div>

<script>
const $=id=>document.getElementById(id);
const v=id=>$(id).value;

async function loadCfg(){
  try{
    const c=await(await fetch('/api/config')).json();
    $('backend').value=c.backend;
    $('rockchip_hw').checked=c.rockchip_hw;
    $('model_file').value=c.model_file||'';
    $('yaml_file').value=c.yaml_file||'';
    $('class_labels').value=(c.class_labels||[]).join(',');
    $('source').value=c.source;
    $('camera_device_id').value=c.camera_device_id;
    $('rtsp_url').value=c.rtsp_url||'';
    $('video_file').value=c.video_file||'';
    $('gain').value=c.gain;
    $('gamma').value=c.gamma;
    $('brightness').value=c.brightness;
    $('resolution_index').value=c.resolution_index;
    $('fps_index').value=c.fps_index;
    $('iq_dir').value=c.iq_dir||'';
    $('conf_threshold').value=c.conf_threshold;
    $('nms_threshold').value=c.nms_threshold;
    $('roi_enabled').checked=c.roi_enabled;
    $('roi_x').value=c.roi?c.roi.x:0;
    $('roi_y').value=c.roi?c.roi.y:0;
    $('roi_w').value=c.roi?c.roi.width:0;
    $('roi_h').value=c.roi?c.roi.height:0;
    $('web_port').value=c.web_port;
    $('debug_logging').checked=c.debug_logging;
    srcVis();
    $('conn').textContent='Connected';
    $('conn').style.color='#34d399';
  }catch(e){
    $('conn').textContent='Error: '+e.message;
    $('conn').style.color='#f87171';
  }
}

function srcVis(){
  const s=+v('source');
  $('r-cam').style.display=s===0?'flex':'none';
  $('r-rtsp').style.display=s===2?'flex':'none';
  $('r-vid').style.display=s===1?'flex':'none';
}

async function post(url,body,stId){
  const el=$(stId);
  el.style.color='#6b7280';
  el.textContent='Saving…';
  try{
    const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(r.ok){el.style.color='#34d399';el.textContent='✓ Saved — restart to apply';}
    else{el.style.color='#f87171';el.textContent='✗ '+r.statusText;}
  }catch(e){el.style.color='#f87171';el.textContent='✗ '+e.message;}
}

function dsai_applyModel(){
  post('/api/config/model',{
    backend:+v('backend'),
    rockchip_hw:$('rockchip_hw').checked,
    model_file:v('model_file'),
    yaml_file:v('yaml_file'),
    class_labels:v('class_labels').split(',').map(s=>s.trim()).filter(Boolean)
  },'st-m');
}
function dsai_applySource(){
  post('/api/config/source',{
    source:+v('source'),
    camera_device_id:+v('camera_device_id'),
    rtsp_url:v('rtsp_url'),
    video_file:v('video_file'),
    gain:+v('gain'),
    gamma:+v('gamma'),
    brightness:+v('brightness'),
    resolution_index:+v('resolution_index'),
    fps_index:+v('fps_index'),
    iq_dir:v('iq_dir')
  },'st-s');
}
function applyDetection(){
  post('/api/config/detection',{
    conf_threshold:+v('conf_threshold'),
    nms_threshold:+v('nms_threshold')
  },'st-d');
}
function dsai_applyRoi(){
  post('/api/config/roi',{
    roi_enabled:$('roi_enabled').checked,
    roi:{x:+v('roi_x'),y:+v('roi_y'),width:+v('roi_w'),height:+v('roi_h')}
  },'st-r');
}
function applySystem(){
  post('/api/config/system',{
    web_port:+v('web_port'),
    debug_logging:$('debug_logging').checked
  },'st-sy');
}
async function doRestart(){
  const el=$('st-x');
  el.style.color='#fbbf24';
  el.textContent='Restarting…';
  try{await fetch('/api/restart',{method:'POST'});}catch(e){}
  el.textContent='Restarting — reconnecting in 6s…';
  setTimeout(loadCfg,6000);
}

loadCfg();
</script>
</body>
</html>
)HTML";

// ── minimal JSON field extractors ─────────────────────────────────────────────
// All operate on the simple flat JSON the browser sends us.

static bool dsai_jStr(const std::string& j, const char* key, std::string& out) {
    std::string pat = std::string("\"") + key + "\"";
    size_t pos = j.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == ':')) ++pos;
    if (pos >= j.size() || j[pos] != '"') return false;
    ++pos;
    out.clear();
    while (pos < j.size() && j[pos] != '"') {
        if (j[pos] == '\\' && pos + 1 < j.size()) { ++pos; out += j[pos]; }
        else out += j[pos];
        ++pos;
    }
    return true;
}

static bool dsai_jInt(const std::string& j, const char* key, int& out) {
    std::string pat = std::string("\"") + key + "\"";
    size_t pos = j.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == ':')) ++pos;
    if (pos >= j.size()) return false;
    bool neg = (j[pos] == '-');
    if (neg) ++pos;
    if (pos >= j.size() || !isdigit((unsigned char)j[pos])) return false;
    out = 0;
    while (pos < j.size() && isdigit((unsigned char)j[pos]))
        out = out * 10 + (j[pos++] - '0');
    if (neg) out = -out;
    return true;
}

static bool jFloat(const std::string& j, const char* key, float& out) {
    std::string pat = std::string("\"") + key + "\"";
    size_t pos = j.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == ':')) ++pos;
    if (pos >= j.size()) return false;
    try { out = std::stof(j.substr(pos)); return true; }
    catch (...) { return false; }
}

static bool dsai_jBool(const std::string& j, const char* key, bool& out) {
    std::string pat = std::string("\"") + key + "\"";
    size_t pos = j.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == ':')) ++pos;
    if (pos + 4 <= j.size() && j.substr(pos, 4) == "true")  { out = true;  return true; }
    if (pos + 5 <= j.size() && j.substr(pos, 5) == "false") { out = false; return true; }
    return false;
}

// Extracts the raw JSON substring of a nested object: "key": { ... }
static bool jObj(const std::string& j, const char* key, std::string& out) {
    std::string pat = std::string("\"") + key + "\"";
    size_t pos = j.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == ':')) ++pos;
    if (pos >= j.size() || j[pos] != '{') return false;
    int depth = 0;
    size_t start = pos;
    while (pos < j.size()) {
        if      (j[pos] == '{') ++depth;
        else if (j[pos] == '}') { if (--depth == 0) { out = j.substr(start, pos - start + 1); return true; } }
        else if (j[pos] == '"') { ++pos; while (pos < j.size() && j[pos] != '"') { if (j[pos] == '\\') ++pos; ++pos; } }
        ++pos;
    }
    return false;
}

static std::vector<std::string> dsai_jStrArr(const std::string& j, const char* key) {
    std::vector<std::string> res;
    std::string pat = std::string("\"") + key + "\"";
    size_t pos = j.find(pat);
    if (pos == std::string::npos) return res;
    pos += pat.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == ':')) ++pos;
    if (pos >= j.size() || j[pos] != '[') return res;
    ++pos;
    while (pos < j.size() && j[pos] != ']') {
        while (pos < j.size() && (j[pos] == ' ' || j[pos] == ',' || j[pos] == '\n' || j[pos] == '\r')) ++pos;
        if (j[pos] == '"') {
            ++pos;
            std::string s;
            while (pos < j.size() && j[pos] != '"') {
                if (j[pos] == '\\' && pos + 1 < j.size()) ++pos;
                s += j[pos++];
            }
            if (pos < j.size()) ++pos;
            if (!s.empty()) res.push_back(s);
        } else if (j[pos] == ']') break;
        else ++pos;
    }
    return res;
}

// ── HTTP response helpers ─────────────────────────────────────────────────────

static std::string httpResp(int code, const char* status,
                             const char* ct, const std::string& body) {
    char hdr[256];
    snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-cache\r\n"
        "\r\n",
        code, status, ct, body.size());
    return std::string(hdr) + body;
}

static std::string dsai_ok200(const std::string& body,
                          const char* ct = "application/json") {
    return httpResp(200, "OK", ct, body);
}
static std::string dsai_err400(const char* msg) {
    std::string b = std::string("{\"error\":\"") + msg + "\"}";
    return httpResp(400, "Bad Request", "application/json", b);
}
static std::string dsai_err404() {
    return httpResp(404, "Not Found", "application/json", "{\"error\":\"not found\"}");
}

// ── WebConfigServer ───────────────────────────────────────────────────────────

WebConfigServer::WebConfigServer(HeadlessApp& app) : app_(app) {}
WebConfigServer::~WebConfigServer() { dsai_stop(); }

bool WebConfigServer::dsai_start() {
    serverFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (serverFd_ < 0) { perror("[WebServer] socket"); return false; }

    int opt = 1;
    setsockopt(serverFd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(app_.dsai_config().webPort));

    if (::bind(serverFd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("[WebServer] bind");
        ::close(serverFd_); serverFd_ = -1;
        return false;
    }
    ::listen(serverFd_, 4);
    running_ = true;
    thread_  = std::thread(&WebConfigServer::serveLoop, this);
    printf("[WebServer] Listening on http://0.0.0.0:%d\n", app_.dsai_config().webPort);
    return true;
}

void WebConfigServer::dsai_stop() {
    running_ = false;
    if (serverFd_ >= 0) {
        ::shutdown(serverFd_, SHUT_RDWR);
        ::close(serverFd_);
        serverFd_ = -1;
    }
    if (thread_.joinable()) thread_.join();
}

void WebConfigServer::serveLoop() {
    while (running_) {
        int fd = ::accept(serverFd_, nullptr, nullptr);
        if (fd < 0) { if (running_) perror("[WebServer] accept"); break; }
        handleClient(fd);
        ::close(fd);
    }
}

void WebConfigServer::handleClient(int fd) {
    std::string req;
    char buf[1024];

    // Read until we have the full header block
    while (req.find("\r\n\r\n") == std::string::npos && req.size() < 8192) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) return;
        req.append(buf, static_cast<size_t>(n));
    }

    // Parse request line: METHOD PATH HTTP/x.x
    size_t lineEnd = req.find("\r\n");
    if (lineEnd == std::string::npos) return;
    std::string line = req.substr(0, lineEnd);
    size_t sp1 = line.find(' ');
    size_t sp2 = line.rfind(' ');
    if (sp1 == std::string::npos || sp1 == sp2) return;
    std::string method = line.substr(0, sp1);
    std::string path   = line.substr(sp1 + 1, sp2 - sp1 - 1);
    if (auto q = path.find('?'); q != std::string::npos) path = path.substr(0, q);

    // Extract Content-Length
    int contentLen = 0;
    for (const char* hdr : { "Content-Length:", "content-length:" }) {
        auto p = req.find(hdr);
        if (p != std::string::npos) {
            contentLen = atoi(req.c_str() + p + strlen(hdr));
            break;
        }
    }

    // Body starts after \r\n\r\n
    size_t bodyStart = req.find("\r\n\r\n");
    if (bodyStart == std::string::npos) return;
    bodyStart += 4;
    std::string body = req.substr(bodyStart);

    // Read any remaining body bytes
    while (static_cast<int>(body.size()) < contentLen && body.size() < 4096) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        body.append(buf, static_cast<size_t>(n));
    }

    std::string resp = dispatch(method, path, body);
    ::send(fd, resp.data(), resp.size(), 0);
}

std::string WebConfigServer::dispatch(const std::string& method,
                                       const std::string& path,
                                       const std::string& body) {
    if (path == "/" && method == "GET")
        return dsai_ok200(kDashboard, "text/html; charset=utf-8");

    if (path == "/api/config" && method == "GET")
        return dsai_ok200(dsai_jsonConfigResp());

    if (path == "/api/config/model"     && method == "POST") return dsai_applyModel(body);
    if (path == "/api/config/source"    && method == "POST") return dsai_applySource(body);
    if (path == "/api/config/detection" && method == "POST") return applyDetection(body);
    if (path == "/api/config/roi"       && method == "POST") return dsai_applyRoi(body);
    if (path == "/api/config/system"    && method == "POST") return applySystem(body);
    if (path == "/api/restart"          && method == "POST") return dsai_triggerRestart();

    if (method == "OPTIONS")
        return httpResp(204, "No Content", "text/plain", "");

    return dsai_err404();
}

// ── GET /api/config ───────────────────────────────────────────────────────────

std::string WebConfigServer::dsai_jsonConfigResp() {
    const AppConfig& c = app_.dsai_config();

    std::string labels = "[";
    for (size_t i = 0; i < c.classLabels.size(); ++i) {
        if (i) labels += ',';
        labels += '"'; labels += c.classLabels[i]; labels += '"';
    }
    labels += ']';

    char buf[2048];
    snprintf(buf, sizeof(buf),
        "{"
        "\"backend\":%d,"
        "\"model_file\":\"%s\","
        "\"yaml_file\":\"%s\","
        "\"class_labels\":%s,"
        "\"source\":%d,"
        "\"camera_device_id\":%d,"
        "\"rtsp_url\":\"%s\","
        "\"video_file\":\"%s\","
        "\"iq_dir\":\"%s\","
        "\"gain\":%d,"
        "\"gamma\":%d,"
        "\"brightness\":%d,"
        "\"resolution_index\":%d,"
        "\"fps_index\":%d,"
        "\"conf_threshold\":%.3f,"
        "\"nms_threshold\":%.3f,"
        "\"roi_enabled\":%s,"
        "\"rockchip_hw\":%s,"
        "\"roi\":{\"x\":%d,\"y\":%d,\"width\":%d,\"height\":%d},"
        "\"web_port\":%d,"
        "\"debug_logging\":%s"
        "}",
        static_cast<int>(c.backend),
        c.modelFile.c_str(), c.yamlFile.c_str(), labels.c_str(),
        static_cast<int>(c.source), c.cameraDeviceId,
        c.rtspUrl.c_str(), c.videoFile.c_str(), c.iqDir.c_str(),
        c.gain, c.gamma, c.brightness,
        c.resolutionIndex, c.fpsIndex,
        c.confThreshold, c.nmsThreshold,
        c.roiEnabled ? "true" : "false",
        c.rockchipHw ? "true" : "false",
        c.roi.x, c.roi.y, c.roi.width, c.roi.height,
        c.webPort,
        c.debugLogging ? "true" : "false"
    );
    return buf;
}

// ── config section handlers ───────────────────────────────────────────────────
// Each handler patches the in-memory config and saves to disk.
// Restart must be triggered separately via POST /api/restart.

static void dsai_saveConfig(HeadlessApp& app) {
    try { app.dsai_config().dsai_saveToFile(app.configPath()); }
    catch (const std::exception& e) {
        fprintf(stderr, "[WebServer] config save: %s\n", e.what());
    }
}

std::string WebConfigServer::dsai_applyModel(const std::string& body) {
    AppConfig& c = app_.dsai_config();
    int ival; std::string s; bool bval;
    if (dsai_jInt(body, "backend",    ival)) c.backend   = static_cast<Backend>(ival);
    if (dsai_jBool(body, "rockchip_hw", bval)) c.rockchipHw = bval;
    if (dsai_jStr(body, "model_file", s))    c.modelFile = s;
    if (dsai_jStr(body, "yaml_file",  s))    c.yamlFile  = s;
    if (body.find("\"class_labels\"") != std::string::npos)
        c.classLabels = dsai_jStrArr(body, "class_labels");
    dsai_saveConfig(app_);
    return dsai_ok200("{\"ok\":true}");
}

std::string WebConfigServer::dsai_applySource(const std::string& body) {
    AppConfig& c = app_.dsai_config();
    int ival; std::string s;
    if (dsai_jInt(body, "source",           ival)) c.source          = static_cast<SourceType>(ival);
    if (dsai_jInt(body, "camera_device_id", ival)) c.cameraDeviceId  = ival;
    if (dsai_jStr(body, "rtsp_url",         s))    c.rtspUrl         = s;
    if (dsai_jStr(body, "video_file",       s))    c.videoFile       = s;
    if (dsai_jStr(body, "iq_dir",           s))    c.iqDir           = s;
    if (dsai_jInt(body, "gain",             ival)) c.gain            = ival;
    if (dsai_jInt(body, "gamma",            ival)) c.gamma           = ival;
    if (dsai_jInt(body, "brightness",       ival)) c.brightness      = ival;
    if (dsai_jInt(body, "resolution_index", ival)) c.resolutionIndex = ival;
    if (dsai_jInt(body, "fps_index",        ival)) c.fpsIndex        = ival;
    dsai_saveConfig(app_);
    return dsai_ok200("{\"ok\":true}");
}

std::string WebConfigServer::applyDetection(const std::string& body) {
    AppConfig& c = app_.dsai_config();
    float fval;
    if (jFloat(body, "conf_threshold", fval)) c.confThreshold = fval;
    if (jFloat(body, "nms_threshold",  fval)) c.nmsThreshold  = fval;
    dsai_saveConfig(app_);
    return dsai_ok200("{\"ok\":true}");
}

std::string WebConfigServer::dsai_applyRoi(const std::string& body) {
    AppConfig& c = app_.dsai_config();
    bool bval;
    if (dsai_jBool(body, "roi_enabled", bval)) c.roiEnabled = bval;
    std::string obj;
    if (jObj(body, "roi", obj)) {
        int v;
        if (dsai_jInt(obj, "x",      v)) c.roi.x      = v;
        if (dsai_jInt(obj, "y",      v)) c.roi.y      = v;
        if (dsai_jInt(obj, "width",  v)) c.roi.width  = v;
        if (dsai_jInt(obj, "height", v)) c.roi.height = v;
    }
    dsai_saveConfig(app_);
    return dsai_ok200("{\"ok\":true}");
}

std::string WebConfigServer::applySystem(const std::string& body) {
    AppConfig& c = app_.dsai_config();
    int ival; bool bval;
    if (dsai_jInt(body,  "web_port",      ival)) c.webPort      = ival;
    if (dsai_jBool(body, "debug_logging", bval)) c.debugLogging = bval;
    dsai_saveConfig(app_);
    return dsai_ok200("{\"ok\":true}");
}

std::string WebConfigServer::dsai_triggerRestart() {
    app_.dsai_requestRestart();
    return dsai_ok200("{\"ok\":true,\"message\":\"restarting\"}");
}
