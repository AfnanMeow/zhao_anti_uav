# Round 3 of P3-H, exactly as pre-registered in ..\PREREG_round3.md.
# Resumable: re-running skips finished work. Launch detached:
#   Start-Process pwsh -ArgumentList '-NoProfile','-File',"<this file>" -WindowStyle Hidden
$ErrorActionPreference = 'Continue'
$root = Resolve-Path "$PSScriptRoot\..\..\.."
$res  = "$PSScriptRoot\..\results"
$py   = "$HOME\Documents\anti_uav\Scripts\python.exe"
$p3h  = "$PSScriptRoot\p3h.py"
$log  = "$res\round3.log"
$env:PYTHONUNBUFFERED = 1
$env:GPU_SOFT_CAP = 540
Set-Location $root
$PID | Set-Content "$res\round3.pid"

function Step([string[]]$a) {
    "`n##### $(Get-Date -Format s)  p3h.py $($a -join ' ')" | Add-Content $log
    & $py $p3h @a *>> $log
}
$modes = 'uniform', 'hetero', 'adaptive'

# R3-1 cosine detector (P9-P12)
foreach ($s in $modes) {
    Step @('--sigma', $s, '--attacks', 'rmd', 'mpelm', 'mpelm_scale', '--defences', 'cos', 'cos_sn', '--tag', 'core')
    Step @('--sigma', $s, '--attacks', 'none', '--defences', 'cos', '--tag', 'core') }
# R3-5 Mix discrepancy: original positional split
Step @('--sigma', 'uniform', '--attacks', 'none', 'mpelm', '--defences', 'mix', '--split', 'positional', '--tag', 'mixsplit')
# R3-2 robust aggregators (P13)
foreach ($s in $modes) {
    Step @('--sigma', $s, '--attacks', 'none', 'rmd', 'mpelm', 'mpelm_scale', '--defences', 'krum', 'median', 'trimmed', '--tag', 'core') }
# R3-4 Arm G clean-window length
foreach ($o in '2', '3', '8') {
    Step @('--sigma', 'hetero', '--attacks', 'rmd', '--defences', 'none', '--onset', $o, '--tag', 'armG') }
Step @('--sigma', 'hetero', '--attacks', 'mpelm_scale', '--defences', 'none', '--onset', '5', '--tag', 'armG')
# R3-3 damage vs sigma spread
foreach ($l in '0.05', '0.1', '0.2', '0.4') {
    Step @('--sigma', "ladder:$l", '--attacks', 'none', '--defences', 'none', '--tag', 'ladder')
    Step @('--sigma', "ladder:$l", '--attacks', 'mpelm_scale', '--defences', 'none', 'norm', 'norm_sn', '--tag', 'ladder') }

"`n##### ROUND 3 COMPLETE $(Get-Date -Format s)" | Add-Content $log
