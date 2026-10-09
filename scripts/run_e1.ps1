# E1 rescue suite (docs/mcts-v8-teacher.md §12.26), Windows desktop.
# screen -> pool (manifest) -> evaluate. Each step resumes from its .jsonl, so re-running the
# script continues where it stopped; a finished step (its .json exists) is skipped.
param(
    [int]$Workers = 14,
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)
$ErrorActionPreference = 'Stop'
Set-Location $Root
$python = Join-Path $Root '.venv-cpu\Scripts\python.exe'
New-Item -ItemType Directory -Force runs\e1 | Out-Null

function Invoke-Step([string]$name, [string[]]$cmdArgs, [string]$output) {
    if (Test-Path $output) { Write-Host "${name}: done earlier ($output)"; return }
    Write-Host "${name}: start $(Get-Date)"
    $proc = Start-Process -FilePath $python -ArgumentList $cmdArgs -NoNewWindow -PassThru -WorkingDirectory $Root `
        -RedirectStandardOutput "runs\e1\$name.log" -RedirectStandardError "runs\e1\$name.err"
    $null = $proc.Handle  # keeps ExitCode readable after exit
    # Priority does not pass to the worker processes: raise every python once a minute.
    while (-not $proc.HasExited) {
        Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.PriorityClass -ne 'High' } | ForEach-Object {
            try { $_.PriorityClass = 'High'; $_.ProcessorAffinity = [IntPtr]0xFFFF } catch {}
        }
        Start-Sleep -Seconds 60
    }
    if ($proc.ExitCode -ne 0) { throw "$name failed (exit $($proc.ExitCode)), see runs\e1\$name.err" }
    Write-Host "${name}: end $(Get-Date)"
}

Invoke-Step 'screen' @('scripts\e1_build_suite.py', 'screen', '--workers', "$Workers",
    '--jsonl', 'runs\e1\screen.jsonl', '--output', 'runs\e1\screen.json') 'runs\e1\screen.json'
Invoke-Step 'pool' @('scripts\e1_build_suite.py', 'pool', '--screen', 'runs\e1\screen.json', '--workers', "$Workers",
    '--jsonl', 'runs\e1\pool.jsonl', '--output', 'runs\e1\e1_manifest.json') 'runs\e1\e1_manifest.json'
Invoke-Step 'evaluate' @('scripts\e1_evaluate.py', '--manifest', 'runs\e1\e1_manifest.json', '--workers', "$Workers",
    '--jsonl', 'runs\e1\eval.jsonl', '--output', 'runs\e1\e1_eval.json') 'runs\e1\e1_eval.json'
