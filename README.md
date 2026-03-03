# WGPU-rs Automata

A WebAssembly cellular automata project built with WGPU and Rust.

## Quick Start

### Prerequisites
- **Rust** (latest stable)
- **Python 3** (for HTTP server)
- **wasm-pack** (install with: `cargo install wasm-pack`)

### Build

**Development build (fast, ~30s):**
```powershell
.\build.ps1 -Profile dev
```

**Release build (optimized, slower):**
```powershell
.\build.ps1 -Profile release
```

Output files go to: `./pkg/`

### Run

**Terminal 1 - Start HTTP server:**
```powershell
python server.py
```

Server runs on: `http://localhost:8080`

**Terminal 2 - Open in browser:**
```
http://localhost:8080/web-interface/main.html
```

## Project Structure

```
.
├── src/
│   ├── main.rs          # Native binary entry point
│   ├── lib.rs           # Library root (re-exports app + state)
│   ├── app.rs           # Application logic + ApplicationHandler
│   └── state.rs         # Game state management
├── web-interface/
│   └── main.html        # Web interface HTML
├── pkg/                 # Generated WASM files (after build)
├── build.ps1            # Build script (dev + release)
├── server.py            # HTTP server for testing
└── Cargo.toml           # Project dependencies
```

## Building for Different Targets

### Native Desktop
```bash
cargo build --release
./target/release/wgpu-rs-automata.exe
```

### WebAssembly (Browser)
```powershell
.\build.ps1 -Profile release
python server.py
# Open http://localhost:8080/web-interface/main.html
```

## Development Workflow

1. **Edit code** in `src/`
2. **Build** with `./build.ps1 -Profile dev` (dev is faster for iteration)
3. **Test** by refreshing browser at `http://localhost:8080/web-interface/main.html`
4. **Release build** when ready: `./build.ps1 -Profile release`

## HTTP Server

The `server.py` script serves files from the project root:

- `http://localhost:8080/web-interface/main.html` → launches WASM app
- `http://localhost:8080/pkg/*` → serves WASM/JS files
- Supports custom port: `python server.py --port 3000`

## Notes

- WASM optimization (wasm-opt) is disabled due to bulk memory compatibility issues
- Rust compiler still applies release optimizations
- Cross-platform support: works on Windows, macOS, Linux
- Both native and web builds from same codebase
