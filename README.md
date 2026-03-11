# gpu-automata

A **GPU cellular automata engine** written in **Rust**  on **WGPU**.
The engine allows defining cellular automata rules programmatically and automatically compiles them into **WGSL shaders** that run on the GPU.

---

## Features
- GPU-accelerated cellular automata simulation
- Written in **Rust** / **WGPU**
- **Runtime shader generation**
- Flexible grid **topologies**
- **Multi-field cells**
- **Rule graphs** for describing automata logic
- **Double buffering** for safe GPU updates
- **Instanced rendering** for fast visualization
- Rendering done in a **vertex shader**
---

## Running

Launch the demo application with:

```bash
cargo run
```

## Automata pattern files located in

/automat_files_for_testing