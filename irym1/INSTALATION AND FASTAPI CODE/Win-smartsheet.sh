# =======================================
# IRYM 1 Helper Script for Windows
# =======================================

# ===== Colors =====
function Write-Color([string]$Text, [string]$Color) {
    Write-Host $Text -ForegroundColor $Color
}

# ===== Project Config =====
$configFile = "$env:USERPROFILE\.irym_config"
$envName = "irym_1"
$envFile = "environment.yml"

# ===== Check Python =====
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Color "Python is not installed. Please install it first." "Red"
    exit
} else {
    Write-Color "Python found." "Green"
}

# ===== Check Conda =====
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Color "Conda not found. Installing Miniconda..." "Yellow"
    $minicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    $tmpInstaller = "$env:TEMP\Miniconda3.exe"
    Invoke-WebRequest $minicondaUrl -OutFile $tmpInstaller
    Start-Process -Wait -FilePath $tmpInstaller -ArgumentList "/S /D=$env:USERPROFILE\Miniconda3"
    $condaPath = "$env:USERPROFILE\Miniconda3\Scripts\conda.exe"
    if (-not (Test-Path $condaPath)) {
        Write-Color "Conda installation failed!" "Red"
        exit
    }
    $env:Path += ";$env:USERPROFILE\Miniconda3\Scripts"
}

# ===== Ask Project Path =====
if (Test-Path $configFile) {
    $PROJECT_DIR = (Get-Content $configFile | Select-String "PROJECT_DIR" | ForEach-Object { ($_ -split "=")[1] })
    if (-not (Test-Path $PROJECT_DIR)) {
        $PROJECT_DIR = Read-Host "Saved project path invalid. Enter correct project path"
        Set-Content $configFile "PROJECT_DIR=$PROJECT_DIR"
    }
} else {
    $PROJECT_DIR = Read-Host "Enter your project path"
    Set-Content $configFile "PROJECT_DIR=$PROJECT_DIR"
}

Set-Location $PROJECT_DIR

# ===== Conda Environment =====
$envs = conda env list | Out-String
if ($envs -match $envName) {
    Write-Color "Activating existing environment: $envName" "Green"
    conda activate $envName
} elseif (Test-Path "$PROJECT_DIR\$envFile") {
    Write-Color "Creating environment $envName from $envFile..." "Yellow"
    conda env create -f "$PROJECT_DIR\$envFile"
    conda activate $envName
} else {
    Write-Color "Environment file not found! Exiting..." "Red"
    exit
}

# ===== Menu =====
while ($true) {
    Write-Color "`n===== IRYM 1 COMMAND CENTER =====" "Cyan"
    Write-Color "[R] Run server" "Blue"
    Write-Color "[M] Make & apply migrations" "Blue"
    Write-Color "[C] Collect static files" "Blue"
    Write-Color "[K] Set NG_KEY in .env" "Blue"
    Write-Color "[0] Clear screen" "Blue"
    Write-Color "[Q] Quit" "Blue"
    $choice = Read-Host "Choose an option"

    switch ($choice.ToUpper()) {
        "R" {
            Write-Color "Running Django server..." "Green"
            python manage.py runserver
        }
        "M" {
            Write-Color "Making migrations..." "Green"
            python manage.py makemigrations
            python manage.py migrate
        }
        "C" {
            Write-Color "Collecting static files..." "Green"
            python manage.py collectstatic --noinput
        }
        "K" {
            $ngKey = Read-Host "Enter NG_KEY"
            $envFilePath = "$PROJECT_DIR\.env"
            if (Test-Path $envFilePath) {
                (Get-Content $envFilePath) -replace "^NG_KEY=.*", "NG_KEY=$ngKey" | Set-Content $envFilePath
                if (-not ((Get-Content $envFilePath) -match "^NG_KEY=")) {
                    Add-Content $envFilePath "NG_KEY=$ngKey"
                }
            } else {
                Set-Content $envFilePath "NG_KEY=$ngKey"
            }
            Write-Color "NG_KEY saved in .env" "Green"
        }
        "0" {
            Clear-Host
        }
        "Q" {
            Write-Color "Exiting..." "Cyan"
            break
        }
        default {
            Write-Color "Invalid option!" "Red"
        }
    }
}

