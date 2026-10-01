# Round 4 of P3-H, exactly as pre-registered in ..\PREREG_round4.md. Resumable.
$ErrorActionPreference = 'Continue'
$root = Resolve-Path "$PSScriptRoot\..\..\.."
$res  = "$PSScriptRoot\..\results"
$py   = "$HOME\Documents\anti_uav\Scripts\python.exe"
$p3h  = "$PSScriptRoot\p3h.py"
$log  = "$res\round4.log"
$env:PYTHONUNBUFFERED = 1
$env:GPU_SOFT_CAP = 540
Set-Location $root
function Step([string[]]$a) {
    "`n##### $(Get-Date -Format s)  p3h.py $($a -join ' ')" | Add-Content $log
    & $py $p3h @a *>> $log
}
$defs = 'none', 'norm_sn', 'cos', 'cos_sn'
Step (@('--sigma', 'hetero', '--attacks', 'mpelm_fixed', '--defences') + $defs + @('--tag', 'round4'))
Step (@('--sigma', 'hetero', '--attacks', 'mpelm', '--onset', '5', '--defences') + $defs + @('--tag', 'round4'))
Step (@('--sigma', 'hetero', '--attacks', 'mpelm_fixed', '--onset', '5', '--defences') + $defs + @('--tag', 'round4'))
"`n##### ROUND 4 COMPLETE $(Get-Date -Format s)" | Add-Content $log
