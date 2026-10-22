<#
.SYNOPSIS
    Render the reproductions to PDF in this directory's own environment.

.DESCRIPTION
    Builds the environment with `uv sync` (eltq and everything else from
    PyPI), points Quarto at its interpreter, and renders
    every page, or the pages named. Everything is derived from the script's
    location, so the render works from any clone.

.PARAMETER Pages
    Pages to render, such as Homer2017a.qmd. Renders them all when omitted.

.EXAMPLE
    .\render.ps1
    .\render.ps1 Homer2017a.qmd
#>
param([string[]]$Pages)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    uv sync --quiet
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }
    $environment = if ($env:UV_PROJECT_ENVIRONMENT) { $env:UV_PROJECT_ENVIRONMENT } else { Join-Path $PSScriptRoot '.venv' }
    $python = if ($IsWindows -or $env:OS -eq 'Windows_NT') { Join-Path $environment 'Scripts\python.exe' } else { Join-Path $environment 'bin/python' }
    $env:QUARTO_PYTHON = $python
    if (-not $Pages) { $Pages = @(Get-ChildItem -Filter *.qmd | ForEach-Object Name) }
    foreach ($page in $Pages) {
        quarto render $page --to house-pdf
        if ($LASTEXITCODE -ne 0) { throw "quarto render failed on $page" }
    }
}
finally {
    Pop-Location
}
