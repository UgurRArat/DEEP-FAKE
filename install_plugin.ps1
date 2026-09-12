$source = "win-spout-extracted\win-spout"
$appDataTarget = "$env:APPDATA\obs-studio\plugins\win-spout"

Write-Host "Installing to AppData: $appDataTarget"
New-Item -ItemType Directory -Path $appDataTarget -Force | Out-Null
Copy-Item -Path "$source\*" -Destination $appDataTarget -Recurse -Force

# Also attempt Program Files if writable
try {
    $pfBin = "C:\Program Files\obs-studio\obs-plugins\64bit"
    $pfData = "C:\Program Files\obs-studio\data\obs-plugins\win-spout"
    if (Test-Path $pfBin) {
        Copy-Item -Path "$source\bin\64bit\*" -Destination $pfBin -Force -ErrorAction Stop
        New-Item -ItemType Directory -Path $pfData -Force | Out-Null
        Copy-Item -Path "$source\data\*" -Destination $pfData -Recurse -Force -ErrorAction Stop
        Write-Host "Successfully installed to Program Files as well!"
    }
} catch {
    Write-Host "Program Files copy skipped (non-admin), AppData installation will be used."
}

Write-Host "Verifying AppData installation:"
Get-ChildItem -Recurse $appDataTarget | Select-Object Name
