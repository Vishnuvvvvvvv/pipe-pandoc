<#
.SYNOPSIS
    Build and run the docx-pipeline Docker container.

.USAGE
    # Build the image
    .\run.ps1 build

    # Start (cheap mode — no AWS cost, default)
    .\run.ps1 start

    # Start with OCR (Textract) enabled
    .\run.ps1 start -Textract

    # Start with both OCR and Chart AI (Bedrock) enabled
    .\run.ps1 start -Textract -Vlm

    # View live logs
    .\run.ps1 logs

    # Stop and remove the container (volumes kept)
    .\run.ps1 stop

    # Rebuild image then start
    .\run.ps1 rebuild
#>

param(
    [Parameter(Position=0, Mandatory=$true)]
    [ValidateSet("build", "start", "stop", "logs", "rebuild", "status")]
    [string]$Command,

    [switch]$Textract,   # pass -Textract to enable USE_TEXTRACT=true
    [switch]$Vlm         # pass -Vlm to enable USE_VLM=true
)

$IMAGE   = "docx-pipeline"
$CONTAINER = "docx-pipeline"
$PORT    = "8001"

function Build-Image {
    Write-Host "Building image '$IMAGE'..." -ForegroundColor Cyan
    docker build -t $IMAGE .
}

function Start-Container {
    # Check if already running
    $existing = docker ps -q -f "name=$CONTAINER"
    if ($existing) {
        Write-Host "Container '$CONTAINER' is already running. Stop it first with: .\run.ps1 stop" -ForegroundColor Yellow
        return
    }

    # Resolve flags
    $useTextract = if ($Textract) { "true" } else { "false" }
    $useVlm      = if ($Vlm)      { "true" } else { "false" }

    Write-Host "Starting '$CONTAINER' — USE_TEXTRACT=$useTextract  USE_VLM=$useVlm" -ForegroundColor Cyan

    # Only the Docling model cache is worth persisting (saves re-downloading ~2GB)
    docker volume create docling_cache | Out-Null

    docker run -d `
        --name $CONTAINER `
        -p "${PORT}:8001" `
        --env-file .env `
        -e "USE_TEXTRACT=$useTextract" `
        -e "USE_VLM=$useVlm" `
        -v "docling_cache:/app/.cache/huggingface" `
        --memory=6g `
        --restart=unless-stopped `
        $IMAGE

    Write-Host ""
    Write-Host "Container started. API available at: http://localhost:$PORT" -ForegroundColor Green
    Write-Host "Docs at:                              http://localhost:$PORT/docs" -ForegroundColor Green
    Write-Host "Run '.\run.ps1 logs' to watch startup logs." -ForegroundColor DarkGray
}

function Stop-Container {
    Write-Host "Stopping '$CONTAINER'..." -ForegroundColor Cyan
    docker stop $CONTAINER 2>$null
    docker rm   $CONTAINER 2>$null
    Write-Host "Stopped. Volumes (uploads + model cache) are preserved." -ForegroundColor Green
}

function Show-Logs {
    Write-Host "Streaming logs for '$CONTAINER' (Ctrl+C to stop watching)..." -ForegroundColor Cyan
    docker logs -f $CONTAINER
}

function Show-Status {
    $running = docker ps --filter "name=$CONTAINER" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    Write-Host $running
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
switch ($Command) {
    "build"   { Build-Image }
    "start"   { Start-Container }
    "stop"    { Stop-Container }
    "logs"    { Show-Logs }
    "status"  { Show-Status }
    "rebuild" { Stop-Container; Build-Image; Start-Container }
}
