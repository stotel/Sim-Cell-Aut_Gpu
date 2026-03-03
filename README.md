# WGPU-rs Automata

A WebAssembly cellular automata project built with WGPU and Rust.

## Quick Start

### Prerequisites
- **Rust** (latest stable)
- **Python 3** (for HTTP server)
- **wasm-pack** (for generating javascript code for WebInterface)

### Build

**Development build:**
```powershell
.\build.ps1 -Profile dev
```

**Release build:**
```powershell
.\build.ps1 -Profile release
```

### Run

**Local:**
```powershell
cargo run
```

**HTTP server:**
```powershell
python server.py
```

Server runs on: `http://localhost:8080`

```
http://localhost:8080/web-interface/main.html
```
