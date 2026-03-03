#!/usr/bin/env pwsh

# WASM Build Script for wgpu-rs-automata

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("dev", "release")]
    [string]$Profile = "dev",
    
    [switch]$Clean
)

function Show-Help {
    Write-Host @"
WASM Build Script
Usage: ./build.ps1 [OPTIONS]

Options:
  -Profile <dev|release>    Build profile (default: dev)
  -Clean                    Clean build artifacts first
  
Examples:
  # Build development (unoptimized, faster)
  ./build.ps1 -Profile dev
  
  # Build release with optimizations
  ./build.ps1 -Profile release
  
  # Clean and rebuild
  ./build.ps1 -Clean -Profile release
"@
}

if ($Clean) {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force pkg -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force target -ErrorAction SilentlyContinue
}

Write-Host "Building WebAssembly ($Profile profile)..." -ForegroundColor Cyan

if ($Profile -eq "dev") {
    # Dev builds use --dev to skip optimization (faster)
    & wasm-pack build --dev --target web --no-opt
} else {
    # Release builds with Rust optimizations, skip wasm-opt
    Write-Host "Building with Rust optimizations (wasm-opt skipped)" -ForegroundColor Yellow
    & wasm-pack build --release --target web --no-opt
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Build successful!" -ForegroundColor Green
    Write-Host "Output available in: ./pkg" -ForegroundColor Green
    Write-Host ""
    Write-Host "To run the web interface:" -ForegroundColor Cyan
    Write-Host "  1. python server.py" -ForegroundColor White
    Write-Host "  2. Open http://localhost:8080/web-interface/main.html" -ForegroundColor White
} else {
    Write-Host "`n[ERROR] Build failed!" -ForegroundColor Red
    exit 1
}
