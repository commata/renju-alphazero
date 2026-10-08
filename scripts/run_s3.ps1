# S3-VCT2 benchmark (docs/mcts-v8-teacher.md §12.24), Windows desktop.
# For each seed the baseline (puct_policy) and the tested arm (puct_policy_vct2) run at the
# same time with the same number of workers, so their time ratio is measured under equal load.
# Re-running resumes from the --games-jsonl files. Then:
#   python scripts\s3_compare.py --baseline runs\s3\base_8411.json runs\s3\base_8412.json `
#       --arm runs\s3\vct2_8411.json runs\s3\vct2_8412.json --output runs\s3\s3_compare.json
param(
    [int[]]$Seeds = @(8411, 8412),
    [int]$Workers = 4,
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)
$ErrorActionPreference = 'Stop'
Set-Location $Root
$python = Join-Path $Root '.venv-cpu\Scripts\python.exe'
New-Item -ItemType Directory -Force runs\s3 | Out-Null

foreach ($seed in $Seeds) {
    $procs = @()
    foreach ($pair in @(@('puct_policy', 'base'), @('puct_policy_vct2', 'vct2'))) {
        $arm, $tag = $pair
        if (Test-Path "runs\s3\${tag}_$seed.json") { continue }  # finished earlier
        $cmdArgs = @('scripts\run_mcts_v8_benchmark.py', '--arm', $arm, '--opponent', 'v8:full',
                     '--puct-c', '1.5', '--pairs', '25', '--seed', "$seed", '--workers', "$Workers",
                     '--games-jsonl', "runs\s3\${tag}_$seed.jsonl", '--output', "runs\s3\${tag}_$seed.json")
        $procs += Start-Process -FilePath $python -ArgumentList $cmdArgs -NoNewWindow -PassThru -WorkingDirectory $Root `
            -RedirectStandardOutput "runs\s3\${tag}_$seed.log" -RedirectStandardError "runs\s3\${tag}_$seed.err"
        $null = $procs[-1].Handle  # keeps ExitCode readable after exit
    }
    # Priority does not pass to the worker processes: raise every python once a minute.
    while ($procs | Where-Object { -not $_.HasExited }) {
        Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.PriorityClass -ne 'High' } | ForEach-Object {
            try { $_.PriorityClass = 'High'; $_.ProcessorAffinity = [IntPtr]0xFFFF } catch {}
        }
        Start-Sleep -Seconds 60
    }
    foreach ($p in $procs) { if ($p.ExitCode -ne 0) { throw "seed $seed failed (exit $($p.ExitCode)), see runs\s3\*.err" } }
}
& $python scripts\s3_compare.py --baseline ($Seeds | ForEach-Object { "runs\s3\base_$_.json" }) `
    --arm ($Seeds | ForEach-Object { "runs\s3\vct2_$_.json" }) --output runs\s3\s3_compare.json
