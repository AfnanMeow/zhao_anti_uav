# Start a P3-H job detached from the calling shell, so no session time limit
# applies. Usage:  .\launch.ps1 <logname> <p3h.py args...>
# Writes results\<logname>.log and results\<logname>.pid. p3h.py resumes
# itself, so re-running the same command after any interruption is safe.
param([Parameter(Mandatory)][string]$Name, [Parameter(ValueFromRemainingArguments)]$Rest)
$root = Resolve-Path "$PSScriptRoot\..\..\.."
$res  = "$PSScriptRoot\..\results"
$py   = "$HOME\Documents\anti_uav\Scripts\python.exe"
$argl = ($Rest | ForEach-Object { "'$_'" }) -join ' '
$cmd  = "`$env:PYTHONUNBUFFERED=1; `$env:GPU_SOFT_CAP=540; Set-Location '$root'; " +
        "& '$py' '$PSScriptRoot\p3h.py' $argl *>> '$res\$Name.log'"
$p = Start-Process pwsh -ArgumentList '-NoProfile', '-Command', $cmd -WindowStyle Hidden -PassThru
$p.Id | Set-Content "$res\$Name.pid"
"started $Name as PID $($p.Id)"
