$k = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
$a = Get-Acl $k
Write-Output ('Owner: ' + $a.Owner)
Write-Output 'Access:'
$a.Access | ForEach-Object {
  Write-Output ('  ' + $_.IdentityReference + ' | ' + $_.RegistryRights + ' | ' + $_.AccessControlType + ' | inherited=' + $_.IsInherited)
}
