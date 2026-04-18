$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$venvSitePackages = Join-Path $root '.venv\Lib\site-packages'
$ipythonDir = Join-Path $root '.ipython'
$jupyterConfigDir = Join-Path $root '.jupyter'
$jupyterDataDir = Join-Path $root '.jupyter-data'
$jupyterRuntimeDir = Join-Path $root '.jupyter-runtime'
$tempDir = Join-Path $root '.temp\tmp'

if (-not (Test-Path $python)) {
    throw "No se encontro el interprete del entorno virtual en: $python"
}

New-Item -ItemType Directory -Force $ipythonDir, $jupyterConfigDir, $jupyterDataDir, $jupyterRuntimeDir, $tempDir | Out-Null

$env:IPYTHONDIR = $ipythonDir
$env:JUPYTER_CONFIG_DIR = $jupyterConfigDir
$env:JUPYTER_DATA_DIR = $jupyterDataDir
$env:JUPYTER_RUNTIME_DIR = $jupyterRuntimeDir
$env:JUPYTER_ALLOW_INSECURE_WRITES = '1'
$env:TEMP = $tempDir
$env:TMP = $tempDir
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$venvSitePackages;$env:PYTHONPATH" } else { $venvSitePackages }

$notebooks = @(
    (Get-ChildItem -LiteralPath (Join-Path $root '01_generacion') -File -Filter *.ipynb | Select-Object -First 1).FullName
    (Get-ChildItem -LiteralPath (Join-Path $root '02_eda') -File -Filter *.ipynb | Select-Object -First 1).FullName
    (Get-ChildItem -LiteralPath (Join-Path $root '03_preparacion') -File -Filter *.ipynb | Select-Object -First 1).FullName
)

foreach ($notebook in $notebooks) {
    if (-not (Test-Path $notebook)) {
        throw "No existe el notebook esperado: $notebook"
    }
}

foreach ($notebook in $notebooks) {
    Write-Host "Ejecutando $notebook" -ForegroundColor Cyan
    & $python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $notebook
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo la ejecucion de $notebook"
    }
}

Write-Host "Flujo completo ejecutado correctamente." -ForegroundColor Green
