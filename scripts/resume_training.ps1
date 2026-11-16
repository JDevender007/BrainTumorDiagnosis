# Resume the paused brain tumor training.
# Run from the project root:  powershell -ExecutionPolicy Bypass -File scripts\resume_training.ps1

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class NativeProcess {
    [DllImport("ntdll.dll")]
    public static extern int NtSuspendProcess(IntPtr processHandle);
    [DllImport("ntdll.dll")]
    public static extern int NtResumeProcess(IntPtr processHandle);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr OpenProcess(uint access, bool inherit, int processId);
    [DllImport("kernel32.dll")]
    public static extern bool CloseHandle(IntPtr handle);
}
"@

$pidFile = "outputs\segmentation_unet3d\training.pid"

$recordedPid = Get-Content $pidFile -ErrorAction SilentlyContinue

$worker = $null

if ($recordedPid) {
    $candidate = Get-CimInstance Win32_Process -Filter "ProcessId = $recordedPid"

    if ($null -ne $candidate -and $candidate.Name -eq "python.exe") {
        $worker = $candidate
    }
}

if ($null -eq $worker) {
    $worker = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.ProcessId -ne $PID } |
        Sort-Object WorkingSetSize -Descending |
        Select-Object -First 1
}

if ($null -eq $worker) {
    Write-Host "No python training process found - nothing to resume."
    exit 1
}

$handle = [NativeProcess]::OpenProcess(0x1F0FFF, $false, $worker.ProcessId)
$result = [NativeProcess]::NtResumeProcess($handle)
[NativeProcess]::CloseHandle($handle)

if ($result -eq 0) {
    Write-Host "Training resumed (PID $($worker.ProcessId))."
    Write-Host "Check progress with: .venv\Scripts\python.exe -m scripts.training_progress"
} else {
    Write-Host "Resume failed with result $result."
    exit 1
}