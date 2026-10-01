# Round 2 of P3-H, exactly as pre-registered in ..\PREREG_round2.md.
# Every step resumes itself, so re-running this whole script after any
# interruption skips finished work. Launch detached:
#   Start-Process pwsh -ArgumentList '-NoProfile','-File',"<this file>" -WindowStyle Hidden
$ErrorActionPreference = 'Continue'
$root = Resolve-Path "$PSScriptRoot\..\..\.."
$res  = "$PSScriptRoot\..\results"
$py   = "$HOME\Documents\anti_uav\Scripts\python.exe"
$p3h  = "$PSScriptRoot\p3h.py"
$log  = "$res\round2.log"
$env:PYTHONUNBUFFERED = 1
$env:GPU_SOFT_CAP = 540
Set-Location $root
$PID | Set-Content "$res\round2.pid"

function Step([string[]]$a) {
    "`n##### $(Get-Date -Format s)  p3h.py $($a -join ' ')" | Add-Content $log
    & $py $p3h @a *>> $log
}

# group 1 reuses the decisive runs (same seeds/keys) without touching that file
if (-not (Test-Path "$res\p3h_core.json")) { Copy-Item "$res\p3h_decisive.json" "$res\p3h_core.json" }

# G4 -- damage under defence, band-exploiting attacker (P8, P8b)
foreach ($s in 'uniform', 'hetero', 'adaptive') {
    Step @('--sigma', $s, '--attacks', 'mpelm_scale', '--defences', 'none', 'norm', 'norm_sn', 'mix', 'mix_sn', '--tag', 'core') }
# G1 -- defence grid (P4) + clean baselines
foreach ($s in 'uniform', 'hetero', 'adaptive') {
    Step @('--sigma', $s, '--attacks', 'none', '--defences', 'none', '--tag', 'core')
    Step @('--sigma', $s, '--attacks', 'rmd', 'mpelm', '--defences', 'none', 'norm', 'norm_sn', 'mix', 'mix_sn', '--tag', 'core') }
# G2 -- Arm G data: onset 5 (P6, P7)
foreach ($s in 'uniform', 'hetero', 'adaptive') {
    Step @('--sigma', $s, '--attacks', 'rmd', '--defences', 'none', '--onset', '5', '--tag', 'armG') }
# G3 -- sigma lying (P5a, P5b)
foreach ($l in '0.8', '0.9', '1.1') {
    Step @('--sigma', 'adaptive', '--attacks', 'rmd', '--defences', 'norm_sn', '--lie', $l, '--tag', 'armF')
    Step @('--sigma', 'adaptive', '--attacks', 'rmd', '--defences', 'norm_sn', '--lie', $l, '--cap', '--tag', 'armF') }
Step @('--sigma', 'adaptive', '--attacks', 'mpelm_scale', '--defences', 'norm_sn', '--lie-track', '--tag', 'armF')
Step @('--sigma', 'adaptive', '--attacks', 'mpelm_scale', '--defences', 'norm_sn', '--lie-track', '--cap', '--tag', 'armF')
# G5 -- controlled sigma ladder (descriptive)
foreach ($s in '0.05', '0.1', '0.2', '0.4') {
    Step @('--sigma', "ladder:$s", '--attacks', 'rmd', '--defences', 'norm', '--tag', 'ladder') }

"`n##### ROUND 2 COMPLETE $(Get-Date -Format s)" | Add-Content $log
