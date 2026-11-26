# CNN und LoRA-Runs nacheinander trainieren.
# Aufruf im Projektordner mit: powershell -ExecutionPolicy Bypass -File .\run_all.ps1
# Logs liegen in experiments/logs.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$logDir = Join-Path $PSScriptRoot "experiments\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$runs = @(
    @{ name = "cnn_resnet50";  args = @("-m", "src.training.train_cnn") },
    @{ name = "vit_lora_r4";   args = @("-m", "src.training.train_vit_lora", "--rank", "4") },
    @{ name = "vit_lora_r8";   args = @("-m", "src.training.train_vit_lora", "--rank", "8") },
    @{ name = "vit_lora_r16";  args = @("-m", "src.training.train_vit_lora", "--rank", "16") }
)

$summary = @()
foreach ($run in $runs) {
    $out = Join-Path $logDir "$($run.name).out.log"
    $err = Join-Path $logDir "$($run.name).err.log"
    $start = Get-Date
    Write-Host "[$($start.ToString('HH:mm'))] Start $($run.name)"
    # Getrennte Logdateien für stdout und stderr schreiben.
    $p = Start-Process -FilePath $py -ArgumentList $run.args -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $min = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
    $status = if ($p.ExitCode -eq 0) { "OK" } else { "FEHLER (Exit $($p.ExitCode))" }
    Write-Host "[$((Get-Date).ToString('HH:mm'))] $($run.name): $status nach $min min"
    $summary += "$($run.name): $status, $min min"
    # Auch nach einem fehlgeschlagenen Lauf fortfahren.
}

$summary | Set-Content -Encoding utf8 (Join-Path $logDir "run_all_summary.txt")
Write-Host "Fertig:"
$summary | ForEach-Object { Write-Host "  $_" }
