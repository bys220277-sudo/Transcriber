# Creates the "Transcriber" shortcut on the desktop (start.bat calls it after the first installation).
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Transcriber.lnk'
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = Join-Path $dir 'start.bat'
$s.WorkingDirectory = $dir
$s.IconLocation = (Join-Path $dir 'transcriber.ico') + ',0'
$s.WindowStyle = 7
$s.Description = 'Transcriber'
$s.Save()
