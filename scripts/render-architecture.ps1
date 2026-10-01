param([string]$BrowserPath)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
    $renderArgs = @('--yes', '--package=@mermaid-js/mermaid-cli@11.12.0', '--', 'mmdc', '-i', 'assets/architecture.mmd', '-b', '#faf8f5', '-w', '1800')
    $configPath = $null
    if ($BrowserPath) {
        $configPath = [System.IO.Path]::GetTempFileName()
        @{ executablePath = $BrowserPath } | ConvertTo-Json | Set-Content -LiteralPath $configPath -Encoding UTF8
        $renderArgs += @('-p', $configPath)
    }
    try {
        foreach ($extension in @('svg', 'png')) {
            & npx.cmd @renderArgs -o "assets/architecture.$extension"
            if ($LASTEXITCODE -ne 0) { throw "Mermaid $extension export failed." }
        }
    } finally {
        if ($configPath) { Remove-Item -LiteralPath $configPath }
    }
} finally {
    Pop-Location
}
